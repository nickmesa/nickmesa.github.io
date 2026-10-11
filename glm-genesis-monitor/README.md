# Pre-Genesis Lightning Monitor

Processing code for the page at **https://nickmesa.github.io/lightning/**, which applies the
object-based GLM lightning analysis from

> Mesa, N., B. Trabing, M. Bell, A. Brammer, P. Duran, P. Klotzbach, K. Musgrave, and S. Stevenson:
> *An object-based analysis of lightning characteristics in pre-tropical cyclogenesis environments.*
> Code and data: https://doi.org/10.5061/dryad.x95x69pzq

to Atlantic and East Pacific disturbances, in real time and for past cases.

- **Real time**: every invest (and every numbered system that has not reached genesis) in the NHC
  real-time b-deck directory, updated hourly.
- **Archive**: finished cases. Developed systems hold the 72 h before genesis; non-developers hold
  the invest's track.

## Where things live in this repository

| Path | What it is |
|---|---|
| `lightning/` | The page: `index.html`, `app.js`, `style.css`, `coast.json`, and synthetic examples in `demo/`. Served by GitHub Pages from `main` like the rest of the site. |
| `glm-genesis-monitor/` | This folder: the Python pipeline, tests and tools. Excluded from the published site in `_config.yml`. |
| `.github/workflows/glm-realtime.yml` | Hourly update. |
| `.github/workflows/glm-archive.yml` | Manual run that adds a past case. |
| branch `glm-data` | The published data: `data/index.json` and `data/storms/<id>.json`. Created by the first workflow run. |

The data sits on its own branch instead of `main` because it is rewritten every hour. On `main`
that would add a commit and a site rebuild each hour and grow the history by roughly a gigabyte
over a season. The `glm-data` branch is folded to one commit per UTC day. The page reads it from
`https://raw.githubusercontent.com/nickmesa/nickmesa.github.io/glm-data/data/` (set in
`lightning/index.html`), which GitHub caches for about five minutes.

```
NHC ATCF b-decks ─┐
                  ├─> pipeline/ (Actions, hourly) ─> glm-data branch: data/index.json, data/storms/<id>.json
GLM L2 LCFA (S3) ─┘                                         │
                                         nickmesa.github.io/lightning/ reads it in the browser
```

| Step | Module | Paper script it follows |
|---|---|---|
| Find disturbances, read tracks, interpolate centers | `pipeline/atcf.py` | `tc_info`, `interp_tc_info` in `mesa_functions.py` |
| List and download LCFA files, keep `group_quality_flag == 0`, bin to 0.05° / 5 min | `pipeline/glm.py` | `timestep_filelist_s3`, `read_glm_filelist_from_s3`, `nc_output.py` |
| Feature detection, ghost-pixel fix, watershed segmentation, per-feature sums, linking | `pipeline/tracking.py` | `tobac_single_day.py`, `tobac_merge_pregenesis.py` |
| Flag cells within 25 km of land, keep features within 200 km | `pipeline/process.py`, `pipeline/land.py` | `tobac_merge_pregenesis.py`, `tracks_200km_72hr.py` |

Every method parameter is in `pipeline/config.py`.

## First run

1. **Actions → GLM real-time update → Run workflow.** The first run creates the `glm-data` branch.
   If several invests are active it may take two or three hourly runs to fill in their history
   (each run downloads at most 1,500 five-minute steps, about 22,000 files). After that it runs on
   its own every hour.
2. Open https://nickmesa.github.io/lightning/. Allow about five minutes after a run for GitHub's
   cache to pick up new data.
3. Add the paper's cases from **Actions → GLM add archive case → Run workflow**:

   | ids | start | end |
   |---|---|---|
   | `al032021,al092021,al062022,al022024` | | |
   | `alA62024` | `2024-06-28T12` | `2024-07-03T12` |

   The second row is AL96 (2024) from 72 h before to 48 h after the 1 July 12Z reference time used
   in the paper. Check the invest id against
   https://ftp.nhc.noaa.gov/atcf/archive/2024/invests/ (NHC replaces the 9 with a letter for each
   reuse of an invest number: `A6` is the first 96L of the season, `B6` the second).

Both workflows ask for `contents: write` themselves, so no repository setting should need to
change. If a run fails at the Publish step with a permission error, set **Settings → Actions →
General → Workflow permissions** to *Read and write permissions*.

To see the page before any data exists, open https://nickmesa.github.io/lightning/?demo=1. That
loads made-up examples from `lightning/demo/` and labels them as synthetic.

## Check these on the first real run

The code was developed without access to NOAA's servers. The tests run it end to end against a
local server that mimics the ATCF site and the GOES buckets (same key layout, same packed netCDF
variables), but nothing has touched a real GLM file or a live b-deck yet. On the first run:

- **Reproduce a paper case.** Run `al092021` (Ida) and compare feature counts, mean group area and
  mean group energy with your 0.05° / 5 min `Tracks_stats_pregenesis_*_200km_72hr.h5`. This is the
  check that matters most. Known sources of small differences are listed below.
- **b-deck parsing.** The Actions log prints one line per disturbance. Confirm the ids, the number
  of steps and the latest positions against the NHC files.
- **Satellite dates** in `config.py` (`EAST_SATS`, `WEST_SATS`). If a bucket has no files for an
  hour the code tries the other satellite at that position, so a wrong date costs a retry, not data.
- **Run time** of the hourly job.

## Differences from the paper workflow

- **Storm-following grid.** The paper gridded one fixed box per storm. Here each 5-minute grid is a
  ±3° box around the center, cut from the same global 0.05° lattice, and feature positions are put
  back on that lattice before linking. Detection, segmentation and statistics are per time step, so
  they are unchanged; features cut by the edge of the box are more than 300 km out.
- **Distance to land** defaults to a 0.05° grid built from Natural Earth 1:10m coastlines
  (`tools/build_land.py`). To use the paper's 0.01° file instead, run
  `python tools/convert_dist2coast.py /path/to/Misc_files/dist2coast_1deg.nc` once and commit the
  `pipeline/data/dist2coast.npz` it writes; the pipeline picks it up automatically and records the
  source in each data file.
- **Real-time centers.** Past the latest best-track fix the center is extrapolated with the motion
  over the previous 12 h (for at most 12 h) and recomputed when the next fix arrives. Earlier fixes
  can also be revised by NHC; each run re-reads the track and reanalyses the whole record.
- **No shear.** The Brammer best-track files carried shear; ATCF b-decks do not, so there is no
  shear-relative view yet.
- **No ABI.** Channel 13 brightness temperature and cold cloud fraction are not included.
- **Genesis** is the first `TS` record in the b-deck, as in `tracks_200km_72hr.py`
  (`GENESIS_TYPES` in `config.py`).
- **East Pacific** disturbances west of 106°W use GOES-West (`WEST_SWITCH_LON`). The paper's cases
  were all GOES-16.

## Run it yourself

```bash
cd glm-genesis-monitor
python -m venv .venv && . .venv/bin/activate
pip install -r requirements-dev.txt

python -m pytest -q                               # unit and end-to-end tests, about a minute
python -m pipeline.run_realtime                   # writes public/data/
python -m pipeline.run_archive --id al092021      # 72 h before Ida's genesis, ~13,000 files
python tools/make_demo.py                         # regenerate the synthetic examples in ../lightning/demo
```

To look at locally produced data, serve the repository root (`python -m http.server` one folder
up) and temporarily point `window.GLM_DATA_BASE` in `lightning/index.html` at
`../glm-genesis-monitor/public/data/`.

A large backfill (a whole season of invests is roughly 400 GB of downloads) is better run on a
cluster with `run_archive`, then committed to the `glm-data` branch under `data/`.

## Page settings

Set these in the `<script>` block before `app.js` in `lightning/index.html`:

| Variable | Purpose |
|---|---|
| `window.GLM_DATA_BASE` | Where `index.json` and `storms/` live. |
| `window.GLM_DOMAINS` | Fixed ranges for the color scale and axes, e.g. `{ area: [60, 3000], energy: [0.5, 500], life: [5, 240] }`, so cases can be compared by eye. By default each record uses its own 2nd to 98th percentile range. |

## Data files

`data/index.json` lists every record. `data/storms/<id>.json` holds one disturbance:

- `track`: best-track fixes (`t` epoch seconds, `lat`, `lon`, `vmax` kt, `mslp` hPa, `type`).
- `steps`: the 5-minute periods (`t0`, `dt`, `n`), the interpolated center (`clat`, `clon`), GLM
  files received per period (`nfiles`, 15 expected, -1 not downloaded yet), the satellite, and the
  first period with an extrapolated center (`extrap_from`).
- `features`: one entry per tobac feature within 200 km, as parallel arrays. `k` period index,
  `lat`/`lon`, `x`/`y` km east/north of the center, `n` groups, `area` mean group area (km²),
  `energy` mean group energy (fJ), `cell` linked cell id, `life` cell lifetime (min), `land` 1 if
  the cell ever came within 25 km of land, `px` pixels in the segmentation mask.
- `method`: the parameters used.

## Housekeeping

- GitHub disables scheduled workflows in repositories with no activity for 60 days and sends an
  email first. Re-enable it from the Actions tab before the season starts.
- Invest numbers are reused within a season, so live invests are stored as `<atcf id>_<MMDD of
  first fix>`, e.g. `al922026_1006`. When an invest is numbered, its record is replaced by the
  numbered system's.
- The older `Lightning Timeseries` workflow in this repository is separate and unchanged.

Not an official forecast product. See https://www.nhc.noaa.gov/ for outlooks and advisories.
