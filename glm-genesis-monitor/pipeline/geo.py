"""Small geometry helpers (same formulas as TC_Functions.py / mesa_functions.py)."""
import numpy as np

EARTH_RADIUS_KM = 6372.8


def haversine(lat1, lon1, lat2, lon2):
    """Great-circle distance in km."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def bearing(lat1, lon1, lat2, lon2):
    """Compass bearing in degrees from point 1 towards point 2."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlon = lon2 - lon1
    theta = np.arctan2(
        np.sin(dlon) * np.cos(lat2),
        np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon),
    )
    return (np.degrees(theta) + 360.0) % 360.0


def offsets_km(clat, clon, lat, lon):
    """East/north offsets (km) and range of points relative to a centre."""
    dist = haversine(clat, clon, lat, lon)
    brg = np.radians(bearing(clat, clon, lat, lon))
    return dist * np.sin(brg), dist * np.cos(brg), dist


def lattice_index(lat, lon, res):
    """Index of the grid cell holding a point on the global lattice.

    Cell centres sit on multiples of `res` and a cell spans centre +/- res/2,
    the same bins GET_GRID() builds in Functions.py.
    """
    gi = np.floor((np.asarray(lat, dtype=np.float64) + 90.0) / res + 0.5).astype(np.int64)
    gj = np.floor((np.asarray(lon, dtype=np.float64) + 180.0) / res + 0.5).astype(np.int64)
    return gi, gj


def lattice_latlon(gi, gj, res):
    return np.asarray(gi) * res - 90.0, np.asarray(gj) * res - 180.0


def destination(clat, clon, x_km, y_km):
    """Latitude and longitude of points x_km east and y_km north of a centre.

    The inverse of offsets_km(): range is measured along the great circle and
    x/y give its compass direction, so an image built with this lines up with
    feature offsets from offsets_km().
    """
    x, y = np.asarray(x_km, dtype=np.float64), np.asarray(y_km, dtype=np.float64)
    d = np.hypot(x, y) / EARTH_RADIUS_KM
    brg = np.arctan2(x, y)
    lat1, lon1 = np.radians(clat), np.radians(clon)
    lat2 = np.arcsin(np.sin(lat1) * np.cos(d) + np.cos(lat1) * np.sin(d) * np.cos(brg))
    lon2 = lon1 + np.arctan2(np.sin(brg) * np.sin(d) * np.cos(lat1), np.cos(d) - np.sin(lat1) * np.sin(lat2))
    return np.degrees(lat2), (np.degrees(lon2) + 540.0) % 360.0 - 180.0


def centred_grid(half_km, res_km):
    """x (east) and y (north) of pixel centres for a square image; row 0 is the northern edge."""
    n = int(round(2 * half_km / res_km))
    edge = (np.arange(n) + 0.5) * res_km - half_km
    return np.meshgrid(edge, -edge)
