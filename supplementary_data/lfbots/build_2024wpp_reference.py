r"""
Builds ``2024wpp_reference.npy``: the AT2024wpp LFBOT reference light curve used for
comparison in ``transient_notebooks/lfbot_end_to_end_simulation.ipynb``.

The photospheric temperature and bolometric luminosity were originally digitized on two
different time grids; this script splines both (in log-log space, since each is a smooth,
wide-dynamic-range function of time) onto a single shared, log-spaced time grid spanning the
overlap of the two source grids, and writes the result as a structured array with fields
``t`` (day), ``T`` (K), and ``L`` (erg/s).

Re-run this script (``python build_2024wpp_reference.py``) if the source data below changes.
"""

from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline

OUT_PATH = Path(__file__).parent / "2024wpp_reference.npy"
N_GRID = 100

# ======================================== #
# Source data (originally hardcoded directly in the notebook)
# ======================================== #
wpp_time_Lbol = np.asarray([2.1082966489608177, 3.5562810827035958, 5.140238833510349, 6.514741447744611, 9.158840678538724, 11.001531162362399, 13.574267107449415, 17.05087748984306, 19.939246127997027, 22.496346037264534, 35.27914867450275, 46.15262831887083, 59.82640682729362, 77.89887268286273, 113.7927876096256])
wpp_Lbol = 10 ** np.asarray([45.21827881784616, 45.142157739789525, 45.11743465460874, 44.97026723703258, 44.71165075389145, 44.44268961142467, 44.11790634504969, 43.772824124526274, 43.49878824352239, 43.31102291764935, 42.65937444088418, 42.38592410663457, 41.731738260600835, 41.564271888876235, 40.959402091703126])

wpp_time_Tbb = np.asarray([2.12350506102289, 3.440659683658525, 5.191402366247135, 6.579586136508955, 9.118498107784243, 11.10537712419451, 13.298237161312215, 17.26059972070161, 20.28297748808043, 22.68685587573649, 27.619273793306355, 36.792220592509125, 44.450405442772976, 61.73543018345186, 80.03503127130507, 114.96446041598476])
wpp_Tbb = np.asarray([41326.94555409202, 35245.322576677616, 32752.42371367296, 31933.190190572528, 31530.263432344284, 27146.297343067527, 25499.57767936435, 22413.47561532735, 20876.376504989137, 20509.19759634709, 19692.533937278113, 18019.430659661622, 19274.50171584991, 12365.416574920364, 15152.577232807324, 12588.435652786315])

# ======================================== #
# Spline both quantities (log-log) onto a shared, log-spaced time grid covering the overlap
# of the two source grids.
# ======================================== #
t_min = max(wpp_time_Lbol.min(), wpp_time_Tbb.min())
t_max = min(wpp_time_Lbol.max(), wpp_time_Tbb.max())
t_grid = np.geomspace(t_min, t_max, N_GRID)

L_spline = CubicSpline(np.log(wpp_time_Lbol), np.log(wpp_Lbol))
T_spline = CubicSpline(np.log(wpp_time_Tbb), np.log(wpp_Tbb))

data = np.zeros(N_GRID, dtype=[("t", "f8"), ("T", "f8"), ("L", "f8")])
data["t"] = t_grid
data["T"] = np.exp(T_spline(np.log(t_grid)))
data["L"] = np.exp(L_spline(np.log(t_grid)))

np.save(OUT_PATH, data)
print(f"Wrote {N_GRID}-point reference to {OUT_PATH}")
