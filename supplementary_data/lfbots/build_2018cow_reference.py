r"""
Builds ``2018cow_reference.npy``: the AT2018cow LFBOT reference light curve used for
comparison in ``transient_notebooks/lfbot_end_to_end_simulation.ipynb``.

The photospheric temperature and bolometric luminosity were originally digitized on two
different time grids; this script splines both (in log-log space, since each is a smooth,
wide-dynamic-range function of time) onto a single shared, log-spaced time grid spanning the
overlap of the two source grids, and writes the result as a structured array with fields
``t`` (day), ``T`` (K), and ``L`` (erg/s).

See ``build_2024wpp_reference.py`` for the same construction applied to AT2024wpp.

Re-run this script (``python build_2018cow_reference.py``) if the source data below changes.
"""

from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline

OUT_PATH = Path(__file__).parent / "2018cow_reference.npy"
N_GRID = 100

# ======================================== #
# Source data (originally hardcoded directly in the notebook)
# ======================================== #
cow_time_Lbol = np.asarray([3.0628990146617685, 3.7484998159419365, 5.2044335083024, 5.961847122479845, 6.530972979705852, 8.27239854643611, 8.94939365657387, 10.051956818070279, 13.736222656067696, 14.338417719993243, 15.45752825914133, 21.738446322992143, 22.170170943819834, 23.360998444136072, 28.59854777422665, 29.905722544617824, 33.770737149338956, 37.5928845624234, 43.712871536585304, 49.983937731507474, 56.85022485184963, 59.875362705866046])
cow_Lbol = 10 ** np.asarray([44.57738375556753, 44.435898890748675, 44.03702232687452, 43.96444154444098, 43.84760418735283, 43.56554228488751, 43.459326505716476, 43.380912933700245, 43.07050412777667, 43.0014638713155, 42.8793157252688, 42.44607790291923, 42.36995659451332, 42.33278107180345, 42.044500553207186, 41.93828477403614, 41.84813753581662, 41.80034043518965, 41.564623109875455, 41.384056285284686, 41.19286788277682, 41.12205736332945])

cow_time_Tbb = np.asarray([3.0897855738544995, 3.6854380874295436, 5.1821099725345885, 6.228196365069224, 6.46147115083082, 8.252521275695583, 9.080428564906978, 10.00977914641733, 13.689715316083308, 14.545995643635836, 15.627273871762538, 20.641589156477245, 22.421963113924413, 24.132468754479323, 27.66882275602075, 29.561398606335135, 32.527806523259564, 35.333387972821505, 42.57778341661202, 50.717474461325935, 57.36460934731182, 59.18583599402263])
cow_Tbb = 10 ** np.asarray([4.512380615493456, 4.477470000816305, 4.394757965769639, 4.423777584283421, 4.380799161927567, 4.312882642649179, 4.311290849228593, 4.321501455742701, 4.252278849555114, 4.247503469293353, 4.218851187722783, 4.210939838371745, 4.197674893200185, 4.220783107942641, 4.135452096541591, 4.16769611711246, 4.246081739272401, 4.163839079208729, 4.259625588419363, 4.141383907920873, 4.201028543440994, 4.187233000462572])

# ======================================== #
# Spline both quantities (log-log) onto a shared, log-spaced time grid covering the overlap
# of the two source grids.
# ======================================== #
t_min = max(cow_time_Lbol.min(), cow_time_Tbb.min())
t_max = min(cow_time_Lbol.max(), cow_time_Tbb.max())
t_grid = np.geomspace(t_min, t_max, N_GRID)

L_spline = CubicSpline(np.log(cow_time_Lbol), np.log(cow_Lbol))
T_spline = CubicSpline(np.log(cow_time_Tbb), np.log(cow_Tbb))

data = np.zeros(N_GRID, dtype=[("t", "f8"), ("T", "f8"), ("L", "f8")])
data["t"] = t_grid
data["T"] = np.exp(T_spline(np.log(t_grid)))
data["L"] = np.exp(L_spline(np.log(t_grid)))

np.save(OUT_PATH, data)
print(f"Wrote {N_GRID}-point reference to {OUT_PATH}")
