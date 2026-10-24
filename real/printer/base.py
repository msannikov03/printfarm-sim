"""Common printer interface, status record and the done-and-cool wait loop."""
from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

log = logging.getLogger("real.printer")

STATES = ("idle", "printing", "paused", "complete", "cancelled", "error", "unknown")


class PrinterError(RuntimeError):
    pass


class DownloadUnsupported(PrinterError):
    """The backend cannot serve the G-code; the message says how to get it by hand."""


class PrintFailed(PrinterError):
    pass


MANUAL_GCODE_HELP = (
    "Get the sliced file by hand instead and pass it with --gcode PATH:\n"
    "  - Creality Print: File > Export G-code (or 'Export plate sliced file') after slicing the same plate;\n"
    "  - or copy it from the USB stick / SD card that was printed from;\n"
    "  - Fluidd/Mainsail (rooted Creality OS): Jobs tab > right-click the file > Download.")


@dataclass
class PrinterStatus:
    state: str = "unknown"                 # one of STATES
    progress: float | None = None          # 0..1
    filename: str | None = None
    bed_c: float | None = None
    bed_target_c: float | None = None
    nozzle_c: float | None = None
    nozzle_target_c: float | None = None
    time_left_s: float | None = None
    elapsed_s: float | None = None
    message: str = ""
    position: list | None = None           # toolhead x, y, z (mm) where the firmware reports it
    backend: str = ""
    t: float = field(default_factory=time.time)
    raw: dict = field(default_factory=dict, repr=False)

    def short(self) -> str:
        def f(v, fmt):
            return "?" if v is None else format(v, fmt)
        pct = None if self.progress is None else 100 * self.progress
        tl = None if self.time_left_s is None else self.time_left_s / 60
        return (f"[{self.backend}] {self.state:<9} {f(pct, '5.1f')}%  bed {f(self.bed_c, '.1f')}/"
                f"{f(self.bed_target_c, '.0f')}C  nozzle {f(self.nozzle_c, '.1f')}/{f(self.nozzle_target_c, '.0f')}C  "
                f"left {f(tl, '.1f')} min  file={self.filename or '?'}" + (f"  msg={self.message}" if self.message else "")
                + (f"  pos={[round(v, 1) for v in self.position]}" if self.position else ""))

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("raw")
        return d


def fnum(v):
    """float or None from numbers / numeric strings ('' and garbage -> None)."""
    if v is None or isinstance(v, bool):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


class Printer(ABC):
    name = "base"

    @abstractmethod
    def status(self) -> PrinterStatus:
        """One fresh status read. Raises PrinterError (or OSError) when the printer is unreachable."""

    def current_file_name(self) -> str | None:
        return self.status().filename

    def download_gcode(self, dest: str | Path, filename: str | None = None) -> Path:
        raise DownloadUnsupported(f"{self.name}: this backend cannot download G-code.\n{MANUAL_GCODE_HELP}")

    def objects(self) -> list[dict] | None:
        """Objects the firmware itself knows (Klipper exclude_object), else None."""
        return None

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()

    def wait_until_done_and_cool(self, release_c: float = 35.0, poll_s: float = 5.0, timeout_s: float | None = None,
                                 max_offline_s: float = 600.0, accept_idle: bool = True,
                                 on_update: Callable[[str, PrinterStatus | None], None] | None = None,
                                 sleep=time.sleep, clock=time.monotonic) -> PrinterStatus:
        return wait_until_done_and_cool(self, release_c, poll_s, timeout_s, max_offline_s, accept_idle,
                                        on_update, sleep, clock)


# phases of the wait loop
WAITING, DONE, COOLING, READY = "waiting_for_print", "done", "cooling", "ready"


def wait_until_done_and_cool(printer: Printer, release_c=35.0, poll_s=5.0, timeout_s=None, max_offline_s=600.0,
                             accept_idle=True, on_update=None, sleep=time.sleep, clock=time.monotonic) -> PrinterStatus:
    """Block until the print has finished AND the bed is at or below release_c.

    State machine: waiting_for_print (printing/paused) -> done (complete seen) -> cooling (bed > release_c) -> ready.
      - 'complete' at the first poll counts as done (the print finished before we connected).
      - 'idle'/'standby' after we saw 'printing' counts as done (Creality OS can drop back to idle quickly).
        'idle' with no print ever seen counts as done only if accept_idle (a warning is logged).
      - 'cancelled' / 'error' raise PrintFailed: never unload a failed plate automatically.
      - Read errors (Wi-Fi drop, printer reboot) are retried every poll; after max_offline_s of continuous
        failures the last error is raised. A bed temperature that is unknown never counts as cool.
    """
    t0 = clock()
    phase, seen_printing, offline_since, last = WAITING, False, None, None

    def emit(ph, st):
        if on_update:
            on_update(ph, st)

    while True:
        if timeout_s is not None and clock() - t0 > timeout_s:
            raise TimeoutError(f"wait_until_done_and_cool: timed out after {timeout_s:.0f} s in phase {phase}")
        try:
            st = printer.status()
            offline_since = None
        except (PrinterError, OSError, TimeoutError, ConnectionError) as e:
            now = clock()
            offline_since = offline_since or now
            log.warning("printer read failed (%s); offline for %.0f s", e, now - offline_since)
            emit("offline", last)
            if now - offline_since > max_offline_s:
                raise
            sleep(poll_s)
            continue
        last = st
        if st.state in ("cancelled", "error"):
            emit("failed", st)
            raise PrintFailed(f"print ended in state '{st.state}' ({st.message or 'no message'}); not unloading")
        if phase in (DONE, COOLING) and st.state == "printing":
            log.warning("a new print started while cooling; waiting for it to finish")
            phase = WAITING
        if phase == WAITING:
            if st.state in ("printing", "paused"):
                seen_printing = True
            elif st.state == "complete":
                phase = DONE
            elif st.state == "idle":
                if seen_printing:
                    phase = DONE
                elif accept_idle:
                    log.warning("printer is idle and no print was seen; treating the bed as finished")
                    phase = DONE
        if phase in (DONE, COOLING):
            if st.bed_c is not None and st.bed_c <= release_c:
                phase = READY
            else:
                phase = COOLING
        emit(phase, st)
        if phase == READY:
            return st
        sleep(poll_s)
