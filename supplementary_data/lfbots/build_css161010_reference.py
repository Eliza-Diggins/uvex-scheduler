r"""
Builds ``css161010_reference.npy``: the CSS161010 LFBOT reference light curve used for
comparison in ``transient_notebooks/lfbot_end_to_end_simulation.ipynb``.

See ``build_2024wpp_reference.py`` for the same construction applied to AT2024wpp.

The photospheric temperature and bolometric luminosity were originally digitized on two
different time grids; this script splines both (in log-log space, since each is a smooth,
wide-dynamic-range function of time) onto a single shared, log-spaced time grid spanning the
overlap of the two source grids, and writes the result as a structured array with fields
``t`` (day), ``T`` (K), and ``L`` (erg/s).

Re-run this script (``python build_css161010_reference.py``) if the source data below changes.
"""

from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline

OUT_PATH = Path(__file__).parent / "css161010_reference.npy"
N_GRID = 100

# ======================================== #
# Source data (originally hardcoded directly in the notebook)
# ======================================== #
css_time_Lbol = np.asarray([1.7519764154145305, 3.8144760236971442, 4.995679803114071, 6.660828557443111, 9.385384910038768, 11.55373384311701, 18.250207778804526, 20.98108821810235, 21.132618695431567, 24.776743568474288, 32.97178855893681, 39.79360345014436, 56.837361198789125, 61.05066764153933])
css_Lbol = 10 ** np.asarray([43.68310590371358, 44.12442338789753, 44.020136741467844, 43.73335413770604, 43.51207126443303, 43.38692728871741, 42.588402507872566, 42.53529461828705, 42.42022752418508, 42.230809384663395, 41.807716530965415, 41.237079066072795, 40.71850548951743, 40.67424891486283])

css_time_Tbb = np.asarray([1.6990963569466964, 3.786108578458578, 4.844137278065901, 6.694596738898933, 9.504212366819777, 11.770357735667945, 17.77193394465908, 20.324597439882375, 23.97795889277987, 31.203521991340068, 40.28922848560961, 55.66545369993913, 60.07396370018612])
css_Tbb = 10 ** np.asarray([4.199300699300699, 4.188797583739218, 4.193396098065358, 4.1918723300046254, 4.193021958586161, 4.173403444804223, 4.1777570678348885, 4.157594351174118, 4.180165165573726, 4.136411254115535, 4.150383663029578, 4.1323433375962555, 4.271135479306685])

# ======================================== #
# Spline both quantities (log-log) onto a shared, log-spaced time grid covering the overlap
# of the two source grids.
# ======================================== #
t_min = max(css_time_Lbol.min(), css_time_Tbb.min())
t_max = min(css_time_Lbol.max(), css_time_Tbb.max())
t_grid = np.geomspace(t_min, t_max, N_GRID)

L_spline = CubicSpline(np.log(css_time_Lbol), np.log(css_Lbol))
T_spline = CubicSpline(np.log(css_time_Tbb), np.log(css_Tbb))

data = np.zeros(N_GRID, dtype=[("t", "f8"), ("T", "f8"), ("L", "f8")])
data["t"] = t_grid
data["T"] = np.exp(T_spline(np.log(t_grid)))
data["L"] = np.exp(L_spline(np.log(t_grid)))

np.save(OUT_PATH, data)
print(f"Wrote {N_GRID}-point reference to {OUT_PATH}")
