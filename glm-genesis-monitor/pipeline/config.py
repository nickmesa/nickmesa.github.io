"""Settings for the GLM pre-genesis lightning monitor.

Method parameters follow the scripts in the Dryad package for
Mesa et al., "An object-based analysis of lightning characteristics in
pre-tropical cyclogenesis environments" (nc_output.py, tobac_single_day.py,
tobac_merge_pregenesis.py, tracks_200km_72hr.py). Change them here only.
"""
import os
from datetime import datetime, timezone

# ---------------------------------------------------------------- method ----
GRID_RES_DEG = 0.05        # lat/lon bin size for GLM groups
TIME_RES_MIN = 5           # accumulation period for one grid
GRID_SPACING_M = 5500.0    # dxy handed to tobac for the 0.05 deg grid
RADIUS_KM = 200.0          # keep features within this distance of the centre
LOOKBACK_H = 72            # pre-genesis analysis window
LAND_KM = 25.0             # cells that ever come this close to land are flagged

# Storm types that mark genesis in a b-deck. The paper scripts use the first
# 'TS' record; 'SS' and 'HU' are only a fallback for unusual best tracks.
GENESIS_TYPES = ("TS",)
GENESIS_FALLBACK_TYPES = ("SS", "HU")

# tobac feature detection (tobac_single_day.py)
FEATURE_KW = dict(
    target="maximum",
    threshold=[1],
    n_min_threshold=0,
    position_threshold="weighted_diff",
    sigma_threshold=0,
    min_distance=0,
)
# tobac segmentation
SEGMENT_KW = dict(target="maximum", method="watershed", threshold=0)
# tobac linking (tobac_merge_pregenesis.py)
LINK_KW = dict(
    method_linking="random",
    adaptive_stop=0.2,
    adaptive_step=0.95,
    subnetwork_size=100,
    d_max=15000,
    stubs=0,
)

# GLM quality control: keep groups with group_quality_flag == 0 only. The
# energy/area QC curve in Get_GLM.py is off, as in nc_output.py (apply_qc=False).

# ------------------------------------------------------------ geometry -----
TOBAC_HALF_DEG = 3.0       # half-width of the storm-following grid given to tobac
CACHE_HALF_DEG = 4.0       # half-width of the gridded cells kept on disk
# A cached step is reused while the (possibly revised) centre stays within
# CACHE_HALF_DEG - TOBAC_HALF_DEG of the centre it was downloaded for.

# ---------------------------------------- environment (archived cases) -----
# Channel 13 brightness temperature at every full-disk scan and hourly ERA5
# relative vorticity, on storm-centred grids. The time series follow
# cold_cloud_fraction() and area_averaged_vort_series() in mesa_functions.py.
ABI_PRODUCT = "ABI-L1b-RadF"       # radiances converted to brightness temperature, as in Get_ABI.py
ABI_CHANNEL = 13
IR_HALF_KM = 384.0                 # half-width of the image
IR_RES_KM = 4.0                    # 192 x 192 pixels
IR_TB_MAX = 330.0                  # stored byte = (IR_TB_MAX - Tb) / IR_TB_STEP, so cold tops are bright
IR_TB_STEP = 0.6
IR_TB_VALID = (150.0, 350.0)       # outside this range is treated as bad data (QC_abi)
IR_MAX_BAD = 0.10                  # more bad pixels than this within the radius: no cold cloud fraction
IR_SHEET_COLS = 8                  # frames are stored side by side in image sheets
IR_SHEET_FRAMES = 64
IR_QUALITY = 70                    # WebP quality of the sheets
CCF_THRESHOLDS_K = (195, 200, 205, 208, 210, 215, 220, 225, 230, 235, 240)
ABI_EAST_VIEW_WEST_LIMIT = -135.0  # ABI sees much farther than GLM; used only when the usual satellite has no scan
ABI_WEST_VIEW_EAST_LIMIT = -85.0

ERA5_URL = os.environ.get("GLM_ERA5_URL", "https://nsf-ncar-era5.s3.amazonaws.com")
ERA5_REGION = (-5.0, 60.0, -180.0, 0.0)     # south, north, west, east kept from each hourly field
VORT_LOW_HPA = (900, 700)          # layer means: every ERA5 level between the bounds, equal weights
VORT_MID_HPA = (600, 400)
VORT_HALF_KM = 384.0
VORT_RES_KM = 24.0                 # 33 x 33 points; ERA5 is about 28 km
VORT_UNIT = 0.5e-5                 # s-1 per stored count (int8)
ENV_WORKERS = int(os.environ.get("GLM_ENV_WORKERS", "12"))   # processes reading ABI files; 0 = read in this process
ERA5_WORKERS = int(os.environ.get("GLM_ERA5_WORKERS", "4"))

# ------------------------------------------------------------ real time ----
LATENCY_MIN = 10           # leave the newest minutes alone; files are still arriving
MAX_EXTRAP_H = 12          # extrapolate the centre this far past the last fix
STALE_H = 18               # an invest with no fix for this long is treated as ended
MAX_LIFETIME_H = 240       # cap on how much of one disturbance is kept
AUTO_ARCHIVE_DAYS = 5      # named storms whose genesis is this recent are archived automatically
BASINS = ("al", "ep")
CHUNK_STEPS = 288          # steps handed to tobac at once (one day)
DOWNLOAD_THREADS = 24
EXPECTED_FILES_PER_STEP = 3 * TIME_RES_MIN   # GLM writes one file every 20 s

# ----------------------------------------------------------- satellites ----
# GOES-East for the Atlantic and the eastern part of the East Pacific,
# GOES-West when the centre is west of WEST_SWITCH_LON (the longitude where
# the two view angles are about equal).
WEST_SWITCH_LON = -106.0
# Where the two views overlap, a 5-minute period with no files from the usual
# satellite is taken from the other one instead (GLM outages can last a day).
# These limits keep that to longitudes well inside the other instrument's view.
WEST_VIEW_EAST_LIMIT = -95.0      # GOES-West is used as a stand-in west of this
EAST_VIEW_WEST_LIMIT = -120.0     # GOES-East is used as a stand-in east of this
STANDIN_AFTER_H = 2               # not for the newest periods, whose files may only be late
EAST_SATS = [  # (operational from, bucket)
    (datetime(2017, 12, 18, tzinfo=timezone.utc), "noaa-goes16"),
    (datetime(2025, 4, 7, 15, tzinfo=timezone.utc), "noaa-goes19"),
]
WEST_SATS = [
    (datetime(2019, 2, 12, tzinfo=timezone.utc), "noaa-goes17"),
    (datetime(2023, 1, 4, 18, tzinfo=timezone.utc), "noaa-goes18"),
]

# ---------------------------------------------------------------- I/O ------
ATCF_URL = os.environ.get("ATCF_URL", "https://ftp.nhc.noaa.gov/atcf").rstrip("/")
# {bucket} is replaced by e.g. noaa-goes19
GLM_URL = os.environ.get("GLM_URL", "https://{bucket}.s3.amazonaws.com").rstrip("/")
GLM_PRODUCT = "GLM-L2-LCFA"

# Optional combined invest + best-track files (BEST_b<id>.nc from the paper's
# Misc_files). An archive run uses one in preference to NHC's b-decks.
TRACK_DIR = os.environ.get("GLM_TRACK_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tracks"))

DATA_DIR = os.environ.get("GLM_DATA_DIR", "public/data")    # published JSON
CACHE_DIR = os.environ.get("GLM_CACHE_DIR", "cache")        # gridded cells, not published
ENV_DIR = os.environ.get("GLM_ENV_DIR", "public-env/env")   # IR sheets and vorticity, published on their own branch
_DATA = os.path.join(os.path.dirname(__file__), "data")
# dist2coast.npz is the paper's 0.01 deg file (tools/convert_dist2coast.py);
# dist2land_0p05.npz is the Natural Earth stand-in shipped with the code.
LAND_FILE = next((p for p in (os.path.join(_DATA, "dist2coast.npz"),) if os.path.exists(p)),
                 os.path.join(_DATA, "dist2land_0p05.npz"))


def method_summary():
    """Parameters written into every product so a file documents itself."""
    return {
        "grid_res_deg": GRID_RES_DEG,
        "time_res_min": TIME_RES_MIN,
        "grid_spacing_m": GRID_SPACING_M,
        "radius_km": RADIUS_KM,
        "lookback_h": LOOKBACK_H,
        "land_km": LAND_KM,
        "feature_threshold": FEATURE_KW["threshold"],
        "link_d_max_m": LINK_KW["d_max"],
        "glm_qc": "group_quality_flag == 0",
        "land_source": "dist2coast 0.01 deg" if LAND_FILE.endswith("dist2coast.npz") else "Natural Earth 1:10m",
    }
