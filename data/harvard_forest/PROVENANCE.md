# Harvard Forest inputs: provenance

Every file here was made outside this repository, in `spatial-lsm-calibration`
(github.com/arob5/spatial-lsm-calibration), and is used as is. Nothing in this repository processes
raw data. Site: Harvard Forest EMS tower (AmeriFlux US-Ha1), PEcAn site 4977, 42.5378 °N,
72.1715 °W.

## Files

### `driver/era5_member_1.clim`

The meteorological driver: ERA5 ensemble member 1, 3-hourly, 2011-12-31 21:00 to 2024-12-31 21:00
UTC (step-start labels), in pySIPNET's 12-column `.clim` layout.

- **Made by** `experiments/single_site_mcmc_vs_eki/run/prepare_drivers.py` (on `main` since commit
  `542058c`), from PEcAn's `ERA5.1.2012-01-01.2024-12-31.clim`
  (MD5 `90c40875ca33261113849e9d08288521`).
- **What the script corrects:** it rebuilds the drifting hour labels from row position; shifts
  labels back 3 h, because accumulated fields cover the preceding interval; replaces snapshot
  variables (temperature, VPD, vapor pressure, wind) by the mean of their two edge values; and
  recomputes soil temperature with PEcAn's 15-day exponential filter run forward in time.
- **`driver/era5_member_1.provenance.json`** is the script's own record of the run: inputs,
  configuration, package versions and the SIPNET binary.
- The corrected file passes pySIPNET's strict time-axis checks (`ClimateDrivers.from_file`).

Another driver can replace this one if it has the same time axis (the observation operator below
refers to its steps).

### `nee_operator.npz`, `nee_features.csv`, `step_axis.json`

The NEE observations and the operator that maps SIPNET's output to them.

- **`step_axis.json`:** SIPNET's output steps for the driver above: 37,992 three-hour steps whose
  end times run from 2012-01-01 00:00 to 2024-12-31 21:00 UTC.
- **`nee_features.csv`:** 400 features, each the mean NEE (µmol CO₂ m⁻² s⁻¹) over the usable day
  (214) or night (186) 3-hour windows of one half-month of one year, 2012–2020.
  - A window is usable when all of its tower records are present and at least two of three hours
    were measured rather than gap-filled (ONEFlux QC flag 0).
  - Day means the sun's elevation exceeds −0.833° throughout the window; night means it stays below.
    Windows containing a sunrise or sunset are dropped.
  - Half-months are days 1–15 and 16 to month end, in local standard time (UTC−5).
  - A feature needs at least four usable windows.
  - Columns: `class`, `year`, `month`, `half`, `n_steps` (usable windows),
    `first_step_end_utc`, `last_step_end_utc`, and `nee` (the observed mean).
- **`nee_operator.npz`:** the sparse 400 × 37,992 matrix A with $A_{fs} = 1/|S_f|$ when step $s$ is
  one of feature $f$'s windows $S_f$, and 0 otherwise. The modeled features are $A x$, where $x$ is
  SIPNET's 3-hourly NEE as a rate in µmol CO₂ m⁻² s⁻¹. Saved with `scipy.sparse.save_npz`.
- **Made by** `operators.build("halfmonth_day_night", steps)` from
  `experiments/single_site_mcmc_vs_eki/exploration/nee_error_model/` on branch
  `feat/hf-nee-error-model` (code last changed in commit `64db1f5`). Its input, the step table
  `output/nee_error_model/steps.pkl`, was made by `data.step_table()` from the processed AmeriFlux
  file `data/processed/net_ecosystem_exchange/ameriflux_nee_hourly_ustar_variable.nc` (series
  `NEE_VUT_REF`). The column renaming above was the only change.

### `sipnet_fixed.csv`

Values for SIPNET parameters that are not calibrated, each with a justification. Columns:
`sipnet_parameter_name` (pySIPNET's name), `sipnet_name`, `units`, `value`, `justification`.

- **Source:** `experiments/single_site_mcmc_vs_eki/model/fixed_sipnet_parameters.csv` on branch
  `feat/harvard-forest-models`, last changed in commit `9e05183`.

## Checksums (SHA-256)

```
84e2324781d63744723818def74b7ae82c35dda34dff6e584149ee91b0db2b53  driver/era5_member_1.clim
ac82684afed7e25825571958a6bca2459e29e7d56380bd768e2db06bb263dbe9  driver/era5_member_1.provenance.json
fbf71a7cb7d10a4f3ad9e6c599d1dc3ebcad9b32d5b2d0b4bdc5886f4bf7735e  nee_features.csv
7ca1f3357dc771ff0903d5cbf549f4bbf371dd01ec0652bfb1c6f210f32dd658  nee_operator.npz
e1fbce99a0e72d54575f59ca0604d1f05d70ab364897c9e4b3e92cbd20116c06  sipnet_fixed.csv
0d47a04cab6f874c66bd67034e790a7510529e753da69aede9cb5afb807692e7  step_axis.json
```
