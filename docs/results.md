# Results: where every number comes from

Every number in the README is produced by a script in this repository. The table below lists, for each
placeholder, the command that produces it, the file it is read from, and how it is computed. All times are
**simulated** seconds, not wall-clock time. Batch runs are seeded, so the same commit and the same seeds give the
same plates.

`scripts/fill_results.py` reads the output files and fills the `{{...}}` placeholders in the README and in `docs/`:

```bash
python scripts/fill_results.py --batch runs/eval --batch-real runs/eval_real --batch-faults runs/eval_faults \
    --des runs/des --farm runs/farm --set PERC_IDEAL_MM="..." ...     # prints the values (dry run)
python scripts/fill_results.py ... --write                               # fills them in
python scripts/fill_results.py --lint                                    # nothing unfilled, no broken links
```

## Unloading batches

| Placeholders | Command | Read from |
|---|---|---|
| `ROWA_*`, `ROWB_*` | `python scripts/eval_unload.py --n 20 --procs 5 --realism realistic --printer 4 --out runs/two_rows_v3/eval_p4` (row A) / `--printer 14 --out runs/two_rows_v3/eval_p14` (row B), default two-row cell | `runs/two_rows_v3/eval_p4/summary.json`, `runs/two_rows_v3/eval_p14/summary.json` |
| `IDEAL_*` | `PRINTFARM_LAYOUT=one_row python scripts/eval_unload.py --n 40 --procs 8 --realism ideal --out runs/final_v3/ideal` | `runs/final_v3/ideal/summary.json` |
| `REAL_*` | same with `--realism realistic --out runs/final_v3/realistic` | `runs/final_v3/realistic/summary.json` |
| `HARSH_*` | same with `--realism harsh --out runs/final_v3/harsh` | `runs/final_v3/harsh/summary.json` |
| `FAULT_*` | same with `--realism realistic --faults --out runs/final_v3/realistic_faults` | `runs/final_v3/realistic_faults/summary.json` |

- `*_PLACED`: `counts["truly_placed"] / counts.parts`, ground truth: the part lies in its intended bin (that is
  what `scripts/fill_results.py` reads). The robot's own verdict is `counts["outcome:placed"]`; the two differ by
  the false successes (`*_FALSE`) and the missed successes (`counts["missed_success"]`: in the bin, but reported
  as a failure). The purge tabs count as parts: one per plate, and each goes to the waste bin.
- `--tote-fill N` (stress option): every tote starts with 0-N parts already in it, so releases land on a pile.
- `*_DOORS`: `door_open_ok / episodes` and `door_close_ok / episodes`.
- `*_BEDS`: `episodes - beds_blocked - debris_left_on_bed`: beds that are truly clear after the unload (ground truth;
  the robot's own camera verdict is reported separately as `bed_not_empty`, which also counts "cannot verify").
- `*_FALSE`: false successes, parts the robot believed placed that are not in their bin (ground truth).
- `*_SPP`: `mean_s_per_placed_part`, the mean simulated duration of one successful pick-and-place (planning
  excluded, door excluded). The door times are in `runs/timings.json`.
- `FAULT_RAKE_SHARE`: `(parts - placed - blocking) / (parts - placed)`, the share of failed parts that did not
  end up blocking the bed: in the reject tray, in the reject bin, pushed off a side edge or dropped. The debris lump is
  not a part: a lump left on the bed shows up only in `*_BEDS` (`debris_left_on_bed`). New batches record where each
  failed part and the lump ended (`end:*`, `debris_end:*` in `summary.json`).
- Fault mode puts one tipped part and one lump of debris on each bed.

The two-row verification table in [sim2real.md](sim2real.md) (written by hand, not placeholders) comes from fresh
seeds on the default cell: `python scripts/eval_unload.py --n 20 --start 100 --realism realistic --faults --printer 4
--out runs/verify2/faults_p4` (and `--printer 14 --out runs/verify2/faults_p14`), and `--n 10 --start 100 --realism
harsh` for the harsh rows; the regression on the published seeds is `--n 20 --start 0 --realism realistic`.

## Perception

| Placeholders | Command | Notes |
|---|---|---|
| `PERC_IDEAL_*` | `python scripts/eval_perception.py --n 150 --n-faults 40 --sweep --procs 3` | perfect rendered depth, robust estimator (the default) |
| `PERC_REAL_*` | same run, `realistic` row | depth with the sensor model in [sim2real.md](sim2real.md) |

The script prints `FOUND n/m; centre error mean X mm, p95 Y mm; yaw error (elongated boxes) mean Z deg`.
Ground-truth poses are used only to compute these errors. The planners never see them.

## Fleet (DES)

| Placeholders | Command | Read from |
|---|---|---|
| `DES_*` | `python scripts/make_timings.py runs/two_rows/eval_both && python -m fleet.des --compare --days 7 --seeds 5 --skus small --out runs/des_final/small` (`eval_both` = the seed files of `eval_p4` and `eval_p14` in one directory, 40 plates; station steps from `runs/station_times.json`) | `runs/des_final/small/kpis_compare.json`: `runs.order_aware.mean` vs `runs.human.mean` (5 seeds) |
| `SENS_*` | `python scripts/des_sensitivity.py runs/final/realistic_faults --skus small --out runs/des_final/sensitivity.json`, copied next to `kpis_compare.json` | `sensitivity.json` rows vs its own human row, in parts per printer per day |
| `DES_ALL_*` | `python -m fleet.des --policy order_aware --skus all --order-scale 1.48 --seed S --out runs/des_final/all148/order_aware_sS` and `--policy human`, S = 0..4 (order rate 1.48 instead of 2 keeps the print load equal to the small mix: large parts add about 35 % print volume per order) | mean of `kpis.json` over the 5 seeds (`runs/des_final/all148/summary.txt`), set by hand |

- `DES_UTIL_*`: `printer_util_pct`, the share of time printers spend printing.
- `DES_PPD_*`: `parts_per_day`.
- `DES_LEAD50_*`, `DES_LEAD_*`: order lead time (arrival to shipping), p50 and p90 (`lead_time_p50_h`, `lead_time_p90_h`).
- `DES_ROBOT_BUSY`: `robot_util_pct`.
- `SENS_BREAKEVEN`: the failure rate per part at which the 20-printer robot cell **without** the rake drops to the
  human baseline's output, linearly interpolated between the simulated rates. Output is compared as parts per
  printer per day, because the human baseline is one operator on a row of 10 printers at the same demand per printer.
- `SENS_PMAX`, `SENS_UTIL_PMAX`: the highest simulated failure rate at which the cell **with** the rake still beats
  that baseline, and its output against the human's.
- `scripts/fill_results.py` computes all three from `sensitivity.json`.
- The DES uses the measured timings in `runs/timings.json`. Every other rate (orders, print times, cooling, operator
  shifts) is an assumption, listed in `fleet/des.py` `Params`.

## Whole cell

| Placeholders | Command | Read from |
|---|---|---|
| `FARM_*` | `python scripts/run_farm.py --hours 12 --plates 16 --realism realistic --video --out runs/farm_box` (default two-row cell; orders draw all SKUs, large parts included) | `runs/farm_box/summary_s0.json` |

- Plates unloaded, `farm_hours`, `picks`, `stats.door_open_ok` / `door_close_ok`, `lead_h_p50` / `lead_h_p90`,
  `stats.interventions` (`FARM_INTERV`) and `station.stats.boxes_complete` / `boxes_sealed` (`FARM_BOXES`).
- `summary_s0.json` does not record the realism preset, and `run_farm.py` defaults to `--realism ideal`: check the
  command when re-running.
- The lead times are censored by the run length: a 3.5 h run ships 24 of its 42 orders, and only those count.

## Bed to box

| Placeholders | Command | Read from |
|---|---|---|
| `BOX_LARGE` | `python scripts/eval_unload.py --n 20 --realism realistic --skus mixed --printer 4 --out runs/verify_v3/mixed_p4` (row A) / `--printer 14 --out runs/verify_v3/mixed_p14` (row B); plates are packed with the farm's 3 h cap | `summary.json` `by_size.large`: `truly_placed / parts` |
| `BOX_BENCH` | `python scripts/run_farm.py --station-bench 12 --realism realistic --out runs/station_bench` (station only: random orders, parts spawned into the totes and the tray) | `runs/station_bench/summary_s0.json` `station.stats`, set by hand |
| `BOX_STATION_S` | the same bench with `--times runs/station_times.json` | `runs/station_times.json` `package`: robot-blocking time per package (box request, large-part places, tote dump and camera check; sealing runs in the background) |
| `BOX_PARKS` | `python -m fleet.des --compare --days 7 --seeds 5 --skus small --queue-policy fifo` vs `--queue-policy package`; with large parts the `DES_ALL_*` runs | `partial_handoffs` per week (an incomplete package's tote or tray slot handed to the station's consolidation shelf, i.e. an operator task): `runs/des_box_pkg/queue_policy.json` (small mix) and `runs/des_final/all148/summary.txt`, set by hand |
| `BOX_TOTE_MAX` | `farm.catalog.TOTE_MAX_PARTS` | set by hand |

- "Large parts" are the housing, the plate and the bar (`farm.catalog.LARGE_SKUS`). Bed to tray is ground truth: the
  part lies in its tray slot after the unload.
- A dense-plate stress run (plates packed up to 8 h, which the farm never prints: `--max-print-h 8`) is reported in
  [sim2real.md](sim2real.md) as a limitation, not as a headline.

## Free-text placeholders (set by hand with `--set KEY=...`)

| Placeholder | What goes there |
|---|---|
| `SIM2REAL_CAMERA`, `SIM2REAL_CALIB` | one-line summaries of the depth-noise and calibration-error models from [sim2real.md](sim2real.md) |
| `REAL_DEMO_LINE` | a link to the real-hardware video, or an empty string to drop the line |

## Reference values at the first public version

These were measured before the realistic-sensor work and are kept here for comparison. Ideal sensors, 40 plates and
195 parts: 195/195 placed, doors 40/40 opened and 40/40 closed, every bed clear, 21.6 s per pick-and-place. Fault batch: 157/195
placed, rake cleared 79 % of the failed parts, 23/40 beds clear. Perception with perfect depth over 10 plates:
48/48 found, centre error 0.67 mm mean (p95 2.13 mm). DES: 80 % vs 48 % printer utilisation, lead time p90 7.6 h vs
74.6 h, robot busy 14 %. Whole cell: 12 plates, 60/60 picks, 12/12 doors.
