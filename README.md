# gasmem: stage cut versus performance in gas membrane separators

`gasmem` is an open Python toolkit for sizing, rating and comparing gas permeation stages. It answers the first question every membrane design starts with:

> **How much of the feed should permeate (the stage cut, θ), and what does that choice cost in purity, recovery, membrane area and compression power?**

It covers the standard model hierarchy, from the well-mixed cell up to multicomponent counter-current hollow-fibre modules with bore-side pressure drop, recycle cascades, energy and indicative cost. Every model is checked against published industrial and academic results (see [Validation](#validation)). Nine Jupyter notebooks explain the engineering with interactive sliders, so the toolkit works as both a design aid and a teaching resource.

The intended reader is a chemical or process engineer. You should be comfortable with mass balances, flash-type stage calculations and compressor duty. You do not need to be a membrane specialist.

---

## Contents

1. [What you can do with it](#what-you-can-do-with-it)
2. [Nomenclature and units](#nomenclature-and-units)
3. [The engineering in one page](#the-engineering-in-one-page)
4. [Choosing a flow model](#choosing-a-flow-model)
5. [Installation](#installation)
6. [Quick start](#quick-start)
7. [Notebooks](#notebooks)
8. [Membrane and feed presets](#membrane-and-feed-presets)
9. [Validation](#validation)
10. [Assumptions and limits](#assumptions-and-limits)
11. [Repository layout](#repository-layout)
12. [Citing, licence, contributing](#citing-licence-contributing)
13. [References](#references)

---

## What you can do with it

| Task | Typical question | Functions |
|---|---|---|
| **Design** (area is free) | What θ and area give 98 % CH₄ in the retentate? | `solve_stage`, `sweep_stage_cut` |
| **Rating** (module is fixed) | My module has 1 200 m² of fibre. What happens at +20 % feed? | `HollowFibreModule`, `rate_module`, `feed_flow_for_stage_cut` |
| **Screening** | Is this separation limited by selectivity or by pressure ratio? | `binary_limits`, `binary_sweep` |
| **Material selection** | Is a more selective but slower membrane worth it? | `robeson`, `Membrane.from_permeability` |
| **Process synthesis** | Single stage, two-stage or three-stage with recycle? | `single_stage`, `two_stage_stripping`, `two_stage_enriching`, `three_stage` |
| **Energy and cost** | kWh and USD per tonne of product | `EnergyAssumptions`, `CostAssumptions`, `specific_energy_kWh_per_tonne`, `specific_cost_per_tonne` |

---

## Nomenclature and units

| Symbol | Meaning | Units used in `gasmem` |
|---|---|---|
| θ | Stage cut: permeate flow divided by feed flow | – |
| x, y | Mole fraction on the high-pressure (feed/retentate) side and the low-pressure (permeate) side | – |
| α | Ideal selectivity: the ratio of the fast-gas permeance to the slow-gas permeance | – |
| p_h, p_l | Feed-side and permeate-side absolute pressure | bar |
| φ | Pressure ratio p_h / p_l | – |
| Q_i | Permeance of component i (pressure-normalised flux) | GPU (input); SI internally |
| P_i | Permeability, Q_i multiplied by the selective-layer thickness | Barrer |
| A | Membrane area | m² |
| F | Molar flow | mol/s (helpers `nm3h_to_mols`, `mols_to_nm3h`) |

Useful conversions:

- 1 GPU = 10⁻⁶ cm³(STP) cm⁻² s⁻¹ cmHg⁻¹ = 3.348 × 10⁻¹⁰ mol m⁻² s⁻¹ Pa⁻¹.
- 1 Barrer = 10⁻¹⁰ cm³(STP) cm cm⁻² s⁻¹ cmHg⁻¹.
- Permeance (GPU) = permeability (Barrer) / thickness (µm), so a 1 Barrer polymer at 0.1 µm gives 10 GPU.
- Normal conditions are 0 °C and 1.01325 bar: 1 Nm³/h = 0.012 39 mol/s.

---

## The engineering in one page

**Local flux.** Each component permeates according to the solution–diffusion law, driven by its partial-pressure difference:

$$J_i = Q_i\,(p_h x_i - p_l y_i)$$

**The two limits.** For a binary, the permeate composition is bounded in two ways. One is the *selectivity limit* (φ → ∞), set by y/(1−y) = α·x/(1−x). The other is the *pressure-ratio limit* (α → ∞), y ≤ φ·x. When φ/α ≪ 1 the separation is **pressure-ratio limited**, and raising selectivity buys almost nothing. Raising the compression ratio, or using vacuum on the permeate, is what helps. `binary_limits` tells you which regime you are in.

**The stage-cut trade-off.** As θ increases:

- the retentate is stripped harder, so retentate purity (slow gas) rises;
- the permeate is diluted by more slow gas, so permeate purity (fast gas) falls;
- the recovery of the slow gas in the retentate falls;
- the area rises steeply, roughly like −ln(1−θ) near full stripping.

No single stage gives both a pure permeate and a pure retentate. That is the reason for recycle cascades, and the reason the toolkit sweeps θ as its basic variable.

**Stage models.** A stage is a 1-D differential contactor, analogous to a heat exchanger whose "duty" is permeation:

- **well-mixed:** both sides at outlet composition, giving a closed-form quadratic for binaries;
- **cross-flow:** the feed flows along the membrane and each element's permeate leaves without mixing (Weller–Steiner);
- **co-current / counter-current:** an initial-value problem and a two-point boundary-value problem respectively, solved by shooting from the closed end of the permeate channel.

**Rating.** For a fixed module, the bore-side pressure profile follows Hagen–Poiseuille flow in the fibre lumen, coupled to the permeation equations by successive substitution.

**Flowsheets.** Recycle loops converge by Wegstein-accelerated successive substitution. Power is calculated for intercooled multistage compression and vacuum pumping.

---

## Choosing a flow model

| Situation | Recommended pattern | Why |
|---|---|---|
| Quick screening, or a spiral-wound element with a short permeate path | `well_mixed` or `cross_flow` | Conservative, and fast |
| Asymmetric hollow fibres with a porous support, where permeate cannot mix axially | `cross_flow` | Pan (1986) showed that cross-flow reproduces commercial cellulose acetate modules |
| Hollow fibres with a dense skin facing the flowing permeate | `countercurrent` (or `cocurrent` if the module is built that way) | Feng et al. (1999) matched their data with the correct flow pattern |
| Design ceiling: the best a single stage can do | `countercurrent` | Highest driving force, so the smallest area for a given duty |
| Fixed module with long or narrow fibres | `rate_module(...)` | Bore pressure drop erodes the effective pressure ratio |

Rule of thumb: in a pressure-ratio-limited separation, counter-current flow can save 20–40 % of the area compared with cross-flow at the same duty. In a selectivity-limited separation the patterns converge.

---

## Installation

You need Python 3.10 or later.

**Option A: pip install from GitHub.**

```bash
pip install "gasmem[notebooks] @ git+https://github.com/SinaGhanbarii/gasmem"
```

**Option B: clone and work in place (recommended if you want the notebooks).**

```bash
git clone https://github.com/SinaGhanbarii/gasmem.git
cd gasmem
python -m pip install -e ".[notebooks,test]"
jupyter lab notebooks/00_start_here.ipynb
```

> **Windows:** if `pip` or `python` is "not recognized", use the launcher instead, for example `py -3.12 -m pip install -e ".[notebooks,test]"` and `py -3.12 -m jupyter lab`.

**Option C: Google Colab (no installation).** In a new Colab notebook, run:

```python
!git clone https://github.com/SinaGhanbarii/gasmem.git
%cd gasmem
!pip install -q ipywidgets
```

Then open any notebook from `gasmem/notebooks/` in the Colab file browser. The first cell of each notebook finds the package itself, whether it is in `/content`, in Google Drive, or in a zip you upload.

**Check the installation:**

```bash
python -m pytest -q        # 75 tests, about a minute
```

---

## Quick start

### 1. One stage at a chosen stage cut

Biogas, 40 % CO₂ / 60 % CH₄, with a polyimide membrane (α = 35), 10 bar feed / 1 bar permeate, 1 000 Nm³/h:

```python
from gasmem import *

m  = Membrane({"CO2": 100, "CH4": 100/35})          # permeances in GPU
xf = {"CO2": 0.40, "CH4": 0.60}

r = solve_stage(m, xf, theta=0.45, p_high=10, p_low=1,
                pattern="countercurrent", feed_flow=nm3h_to_mols(1000))
print(r.summary())
```

```
countercurrent: stage cut = 0.4500, area = 1267 m2, p_h/p_l = 10/1 bar
  comp     feed  retent.    perm.  rec->perm
   CO2   0.4000   0.0497   0.8282     0.9317
   CH4   0.6000   0.9503   0.1718     0.1289
```

### 2. Design to a product specification

Find the stage cut that gives 98 % CH₄ in the retentate:

```python
from scipy.optimize import brentq

f  = lambda th: solve_stage(m, xf, th, 10, 1, "countercurrent").x_ret[1] - 0.98
th = brentq(f, 0.3, 0.7)
r  = solve_stage(m, xf, th, 10, 1, "countercurrent", nm3h_to_mols(1000))
print(f"theta = {th:.3f}, CH4 recovery = {r.recovery('CH4', 'retentate'):.3f}, area = {r.area:.0f} m2")
# theta = 0.500, CH4 recovery = 0.817, area = 1744 m2
```

A single stage loses about 18 % of the methane to the permeate. That loss is what drives the cascade in example 5.

### 3. Rate an existing module, including bore pressure drop

```python
mod = HollowFibreModule.for_area(r.area, length=1.0, d_inner=200e-6, d_outer=400e-6)
rr  = rate_module(m, xf, nm3h_to_mols(1200), 10, 1, mod, "countercurrent")
print(f"theta = {rr.theta:.3f}, CH4 = {rr.x_ret[1]:.4f}, bore dp = {rr.extra['bore_dp_bar']:.3f} bar")
# theta = 0.467, CH4 = 0.9586, bore dp = 0.126 bar
```

Pushing 20 % more gas through the same module lowers the stage cut and drops the product off specification. This is the turndown and overload behaviour an operator sees.

### 4. Sweep the stage cut for all flow patterns

```python
df = sweep_stage_cut(m, xf, p_high=10, p_low=1)   # tidy DataFrame: pattern, theta, area, compositions, recoveries
```

The same study is available from the command line:

```bash
python scripts/stage_cut_sweep.py --help
```

### 5. Compare cascades on purity, recovery, area and power

```python
fs = two_stage_stripping(m, xf, nm3h_to_mols(1000),
                         StageSpec(0.33, 10, 1, "cross_flow"),
                         StageSpec(0.335, 10, 1, "cross_flow"))
print(fs.summary())
```

```
two-stage stripping: converged in 17 iterations; recycle/feed = 0.289
  total area 2778 m2, total power 138.9 kW
  slow-gas product (retentate 2): 7.12 mol/s; CO2 0.0201, CH4 0.9799
```

Recycling the second-stage permeate raises the CH₄ recovery from about 82 % to about 98 %, at the cost of about 60 % more area and about 30 % more compression. `python scripts/multistage_biogas.py` compares the single-, two- and three-stage layouts side by side.

### 6. Screening: which limit am I against?

```python
binary_limits(alpha=35, phi=10, x=0.4)
# {'y_selectivity_limit': 0.959, 'y_pressure_ratio_limit': 1.0, 'y_local': 0.947,
#  'phi_over_alpha': 0.286, 'regime': 'pressure-ratio limited'}
```

---

## Notebooks

Each notebook starts from a process question and ends with a design takeaway. They are interactive (ipywidgets) where it helps.

| # | Notebook | Engineering content |
|---|---|---|
| 00 | `start_here` | Concepts, the toolkit in five minutes, and which notebook to read next |
| 01 | `well_mixed` | The closed-form binary stage, the pressure-ratio and selectivity limits, and a θ slider |
| 02 | `cross_flow` | The Weller–Steiner check, and a comparison with well-mixed |
| 03 | `co_and_counter_current` | Shooting the counter-current boundary-value problem, axial profiles, and area savings |
| 04 | `multicomponent` | Air with water vapour, refinery H₂ purge, and the behaviour of intermediate components |
| 05 | `module_rating_pressure_drop` | A fixed module, bore pressure drop, turndown, and fibre diameter and length |
| 06 | `multistage_energy_cost` | Recycle cascades, compressor and vacuum power, and an indicative cost per tonne |
| 07 | `literature_validation` | Reproduction of Merkel, Coker, Pan and Feng operating points |
| 08 | `experimental_data_point_by_point` | Digitised published curves compared point by point, with error statistics |

---

## Membrane and feed presets

The presets are taken from the cited papers so that the examples have realistic numbers. Use `describe()` to print the full source information for any preset.

```python
from gasmem import MEMBRANE_PRESETS, FEED_PRESETS, PAIRINGS, describe
describe("Polyimide CO2/CH4, a=35 (Makaruk 2010)")
```

| Membrane preset | Typical service | Source |
|---|---|---|
| Polaris CO2/N2 (Merkel 2010) | Post-combustion CO₂ capture | Merkel et al., 2010 |
| Polaris pure-gas (MTR 2012) | CO₂ capture, full permeance set | MTR / DOE report, 2012 |
| Polyimide CO2/CH4, a=35 (Makaruk 2010) | Biogas upgrading | Makaruk et al., 2010 |
| Cellulose acetate H2 recovery (Pan 1986) | H₂ recovery from ammonia purge gas | Pan, 1986 |
| Cellulose acetate CO2/hydrocarbons (Pan 1986) | Acid-gas removal | Pan, 1986 |
| Cellulose acetate H2S/CO2 sour gas (Pan 1986) | Sour-gas treating | Pan, 1986 |
| Polysulfone-like air (Coker 1998) | Air separation and dehydration | Coker et al., 1998 |
| Polysulfone-like H2/hydrocarbons (Coker 1998) | Refinery H₂ recovery | Coker et al., 1998 |
| Asymmetric hollow fibre air (Feng 1999) | N₂ generation | Feng et al., 1999 |

The matching feed presets are flue gas with 13 % CO₂, biogas with 60 % CH₄, ammonia purge gas, acid gas with hydrocarbons, sour gas with 22 % H₂S, humid air at 40 °C and 10 atm, refinery hydrotreater purge, and air. `PAIRINGS` links each membrane to its feed.

You can define your own membrane from permeances in GPU, or from permeabilities and a skin thickness:

```python
m = Membrane.from_permeability({"O2": 1.4, "N2": 0.23}, thickness_um=0.1)   # Barrer and µm, giving GPU
```

---

## Validation

### Published operating points

| Case | Quantity | Paper | `gasmem` |
|---|---|---|---|
| Merkel et al. (2010), Fig. 6, CO₂ capture | Cross-flow stage cut / area | 28.9 % / 11.0 million m² | 28.9 % / 11.0 million m² |
| | Counter-current stage cut | 40.6 % | 40.7 % |
| Coker et al. (1998), air dehydration | Stage cut | 94.6 % | 94.65 % |
| Weller & Steiner (1950) | Cross-flow analytical solution | exact | matches to solver tolerance |

### Digitised data, point by point

These datasets ship with the package and load with `gasmem.datasets.load("Pan1986")` and similar calls.

| Source | Data | Model | Agreement |
|---|---|---|---|
| Pan (1986), Figs. 2–3 | H₂ and CO₂ separation, commercial cellulose acetate fibres | Cross-flow | RMS 0.2–1.5 % |
| Pan (1986), Figs. 6–7 | Field data, with fitted permeances | Cross-flow | RMS 0.1–1 % |
| Feng et al. (1999), Figs. 4–5 | N₂ purity and recovery, bore-side feed | The flow pattern used in the experiment | RMS 0.1–0.4 % |
| Feng et al. (1999), Figs. 13–14 | N₂ generation at 23 °C and 45 °C | Counter-current | Good below 97.5 % N₂; above that, the model over-predicts, as Feng's own model also does |
| Makaruk et al. (2010), Figs. 15–16 | Three-stage biogas upgrading | Recycle cascade | Area RMS 3.4 %, power about 7 %; same least-area selectivities (about 22 and 12) |

`pytest` re-runs all of these comparisons, so an accidental regression in a solver fails the test suite.

---

## Assumptions and limits

The models make the following assumptions:

- **Solution–diffusion with constant permeances.** There is no pressure or concentration dependence, plasticisation, or competitive sorption. For CO₂ in glassy polymers above about 10–15 bar CO₂ partial pressure, use permeances measured with mixed gas at your conditions.
- **Isothermal operation.** Joule–Thomson cooling across the membrane is not modelled. Temperature enters only through the permeances you supply.
- **Ideal gas, plug flow,** and no concentration polarisation. This is why the model over-predicts Feng's data at very high N₂ purity.
- **Shell-side pressure drop is neglected.** Bore-side pressure drop is modelled in rating mode only.
- **Shell-side feed in Feng's modules.** The model shows a residual offset of about 0.4 points that bore pressure drop does not explain. It is most likely caused by non-ideal shell-side flow.
- **Pan's recovery data** sit about 3 points above the recovery implied by the paper's own mass balance. We report the comparison as published.
- **Costs are indicative.** `CostAssumptions` defaults are placeholders for comparing options. They are not an estimate. Replace them with your own membrane price, replacement interval, electricity price, and capital and annualisation factors before quoting any number.

---

## Repository layout

```
gasmem/
├── gasmem/                 the package
│   ├── core.py             Membrane, StageResult, local permeate solver, unit helpers
│   ├── wellmixed.py        well-mixed stage (binary closed form, multicomponent)
│   ├── crossflow.py        cross-flow stage (ODE in retentate flow)
│   ├── hollowfiber.py      co-current and counter-current (shooting BVP)
│   ├── module.py           HollowFibreModule, rating with bore pressure drop
│   ├── stagecut.py         solve_stage, sweep_stage_cut, binary_limits
│   ├── flowsheets.py       single, two- and three-stage cascades with recycle
│   ├── energy.py           compression, vacuum, specific energy and cost
│   ├── robeson.py          Robeson upper bounds (1991, 2008)
│   ├── properties.py       literature-based membrane and feed presets
│   ├── datasets.py         loader for the digitised validation data
│   └── data/*.csv          digitised data from Pan, Feng and Makaruk
├── notebooks/              00–08 interactive notebooks
├── scripts/                command-line studies (stage-cut sweep, biogas cascades)
├── tests/                  model and literature regression tests
└── tools/                  notebook builder and digitisation records
```

The notebooks are generated from `tools/build_notebooks.py`. To change a notebook, edit the builder and run `python tools/build_notebooks.py --run 03` (or omit the number to rebuild all of them).

---

## Citing, licence, contributing

If `gasmem` helps your work, please cite it:

> Ghanbari, M. S. (2026). *gasmem: stage cut versus performance in gas membrane separators* (v1.1.0). https://github.com/SinaGhanbarii/gasmem

Please also cite the original papers behind any preset or dataset you use.

The code is released under the **MIT licence** (see [LICENSE](LICENSE)). Digitised data points are approximate readings of published figures, provided for model validation. The original figures remain the property of their publishers.

Contributions are welcome, particularly:

- validated permeance sets for additional membranes;
- further published datasets;
- non-isothermal or concentration-polarisation extensions;
- cost correlations with documented sources.

Please open an issue first, and include a reference for any new data.

---

## References

1. Coker, D. T., Freeman, B. D., Fleming, G. K. (1998). Modeling multicomponent gas separation using hollow-fiber membrane contactors. *AIChE Journal* 44, 1289–1302.
2. Feng, X., Ivory, J., Rajan, V. S. V. (1999). Air separation by integrally asymmetric hollow-fiber membranes. *AIChE Journal* 45, 2142–2152. doi:10.1002/aic.690451013
3. Hosseini, S. S., et al. (2015). Comparative study of gas separation membrane modules: modelling approaches. *(see `gasmem/properties.py` for the full record)*
4. Makaruk, A., Miltner, M., Harasek, M. (2010). Membrane biogas upgrading processes for the production of natural gas substitute. *Separation and Purification Technology* 74, 83–92.
5. Merkel, T. C., Lin, H., Wei, X., Baker, R. (2010). Power plant post-combustion carbon dioxide capture: an opportunity for membranes. *Journal of Membrane Science* 359, 126–139.
6. Membrane Technology and Research, Inc. (2012). *Pilot testing of a membrane system for post-combustion CO₂ capture.* US DOE/NETL final report.
7. Pan, C. Y. (1986). Gas separation by high-flux, asymmetric hollow-fiber membrane. *AIChE Journal* 32, 2020–2027.
8. Robeson, L. M. (1991). Correlation of separation factor versus permeability for polymeric membranes. *Journal of Membrane Science* 62, 165–185.
9. Robeson, L. M. (2008). The upper bound revisited. *Journal of Membrane Science* 320, 390–400.
10. Sidhoum, M., Sengupta, A., Sirkar, K. K. (1988). Asymmetric cellulose acetate hollow fibers: studies in gas permeation. *AIChE Journal* 34, 417–425.
11. Weller, S., Steiner, W. A. (1950). Engineering aspects of separation of gases: fractional permeation through membranes. *Chemical Engineering Progress* 46, 585–590.
