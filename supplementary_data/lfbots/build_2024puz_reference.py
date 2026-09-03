r"""
Builds ``2024puz_reference.npy``: the AT2024puz LFBOT reference light curve used for
comparison in ``transient_notebooks/lfbot_end_to_end_simulation.ipynb``.

The photospheric temperature and bolometric luminosity were originally digitized on two
different time grids; this script splines both (in log-log space, since each is a smooth,
wide-dynamic-range function of time) onto a single shared, log-spaced time grid spanning the
overlap of the two source grids, and writes the result as a structured array with fields
``t`` (day), ``T`` (K), and ``L`` (erg/s).

See ``build_2024wpp_reference.py`` for the same construction applied to AT2024wpp.

Re-run this script (``python build_2024puz_reference.py``) if the source data below changes.
"""

from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline

OUT_PATH = Path(__file__).parent / "2024puz_reference.npy"
N_GRID = 100

# ======================================== #
# Source data (originally hardcoded directly in the notebook)
# ======================================== #
puz_time_Lbol = np.asarray([2.967247744237967, 4.442993477467015, 5.8616469352714855, 7.422586759016108, 9.022395638895775, 10.464889422946804, 11.587511335185038, 12.424252535082804, 14.749803710446598, 15.56247346993425, 20.979189110308564, 22.57468097680877, 23.818477905912324, 31.433705313042076, 56.65372905612643])
puz_Lbol = 10 ** np.asarray([44.70674912763483, 44.733802377372406, 44.74111038611024, 44.85585974070187, 44.89403387330137, 44.77542625322704, 44.79694175721297, 44.749144656586004, 44.36853811455644, 44.52273255978893, 44.15242985616613, 43.90813356407274, 43.81430962580499, 43.62032965474198, 42.742869301256775])

puz_time_Tbb = np.asarray([2.9135890458731404, 4.497539697238392, 5.882331959189844, 7.333881456927652, 8.501456449797617, 10.142973490575558, 11.633919080618215, 12.338904968419534, 14.560898101294981, 15.191234598918536, 20.309467067336655, 22.65063075552319, 23.97907496348101, 30.847562213491504, 54.67810370691144])
puz_Tbb = 10 ** np.asarray([4.371989877826453, 4.337535033060325, 4.30782155587603, 4.3717585916029496, 4.424423825202036, 4.383050801338739, 4.433566433566433, 4.418179097167424, 4.298610922151778, 4.396159288182634, 4.346847704824358, 4.273094609670485, 4.253462490816577, 4.317936926886343, 4.2882302522380344])

# ======================================== #
# Spline both quantities (log-log) onto a shared, log-spaced time grid covering the overlap
# of the two source grids.
# ======================================== #
t_min = max(puz_time_Lbol.min(), puz_time_Tbb.min())
t_max = min(puz_time_Lbol.max(), puz_time_Tbb.max())
t_grid = np.geomspace(t_min, t_max, N_GRID)

L_spline = CubicSpline(np.log(puz_time_Lbol), np.log(puz_Lbol))
T_spline = CubicSpline(np.log(puz_time_Tbb), np.log(puz_Tbb))

data = np.zeros(N_GRID, dtype=[("t", "f8"), ("T", "f8"), ("L", "f8")])
data["t"] = t_grid
data["T"] = np.exp(T_spline(np.log(t_grid)))
data["L"] = np.exp(L_spline(np.log(t_grid)))

np.save(OUT_PATH, data)
print(f"Wrote {N_GRID}-point reference to {OUT_PATH}")
