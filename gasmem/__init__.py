"""gasmem - stage cut versus performance in gas membrane separators.

Models
------
well_mixed       complete mixing, binary closed form + multicomponent
cross_flow       numerical (any n) + Weller-Steiner closed form (binary)
cocurrent        plug flow, permeate with the feed (IVP)
countercurrent   plug flow, permeate against the feed (BVP by shooting)
rate_module      fixed hollow-fibre module, stage cut as output, bore pressure drop
flowsheets       single, two-stage and three-stage layouts with recycle,
                 compression/vacuum energy and an illustrative cost model

Units in the public API: bar, mol/s, GPU, m2, K.
"""
from .core import InfeasibleError, Membrane, StageResult, local_permeate
from .crossflow import cross_flow, weller_steiner, weller_steiner_theta
from .energy import (CostAssumptions, EnergyAssumptions, annual_cost, compression_power,
                     stage_economics, stage_energy)
from .flowsheets import (FlowsheetResult, StageSpec, single_stage, specific_cost_per_tonne,
                         specific_energy_kWh_per_tonne, three_stage, two_stage_enriching,
                         two_stage_stripping)
from .hollowfiber import cocurrent, countercurrent
from .module import HollowFibreModule, feed_flow_for_stage_cut, rate_module
from .properties import (ATM_BAR, BAR, BARRER, FEED_INFO, FEED_PRESETS, GPU, LITERATURE,
                         MEMBRANE_INFO, MEMBRANE_PRESETS, PAIRINGS, barrer_to_gpu, describe,
                         mols_to_nm3h, nm3h_to_mols)
from .robeson import (UPPER_BOUND_1991, UPPER_BOUND_2008, distance_to_bound,
                      upper_bound_selectivity)
from .stagecut import (LABELS, PATTERNS, binary_limits, binary_sweep, solve_stage,
                       sweep_stage_cut)
from .wellmixed import well_mixed, well_mixed_binary

__version__ = "1.1.0"
