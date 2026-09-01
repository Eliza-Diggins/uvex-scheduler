"""
Composite supernova SED models.
"""

__all__ = [
    "TopHatCCSNeSED",
    "TypeIIPExcessSED",
    "TypeIIPSED",
    "VillarCoolingBlackbodySED",
]

from ._IIp_excess import TypeIIPExcessSED, TypeIIPSED
from ._tophat import TopHatCCSNeSED
from ._villar import VillarCoolingBlackbodySED
