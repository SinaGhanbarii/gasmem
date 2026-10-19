# gasmem: stage cut versus performance in gas membrane separators

[![tests](https://github.com/SinaGhanbarii/gasmem/actions/workflows/tests.yml/badge.svg)](https://github.com/SinaGhanbarii/gasmem/actions/workflows/tests.yml) [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE) ![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)

`gasmem` is an open Python toolkit for sizing, rating and comparing gas permeation stages. It answers the first question every membrane design starts with:

> **How much of the feed should permeate (the stage cut, θ), and what does that choice cost in purity, recovery, membrane area and compression power?**

It covers the standard model hierarchy, from the well-mixed cell up to multicomponent counter-current hollow-fibre modules with permeate sweep and bore- and shell-side pressure drop, recycle cascades, energy and indicative cost, and an optimiser that chooses stage cuts and pressures for a product target. Every model is checked against published industrial and academic results (see [Validation](#validation)), and you can check it against your own data in a few lines. Twelve Jupyter notebooks explain the engineering with interactive sliders, and each opens in Google Colab with one click, so the toolkit works as both a design aid and a teaching resource.

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
| **Optimisation** | Which stage cuts and feed pressure give the cheapest 98 % / 97 % biogas plant? | `stage_cut_for_spec`, `optimise` |
| **Sweep gas** | How much does an air or N₂ sweep cut the area of a CO₂-capture stage? | `solve_stage(..., sweep_flow=...)`, `StageSpec(..., sweep_ratio=...)` |
| **Your own data** | Does the model reproduce my pilot module? What are its effective permeances? | `gasmem.userdata`: `template`, `compare`, `fit_permeances` |
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

**Sweep.** A sweep gas entering the permeate channel at its closed end starts the permeate side at P = S·z instead of zero, which lowers the permeate-side partial pressure of the fast gas where it matters most. The stage cut keeps its meaning, the fraction of the feed that crosses the membrane.

**Rating.** For a fixed module, the bore-side pressure profile follows Hagen–Poiseuille flow in the fibre lumen. Optionally, the shell-side (feed) pressure drop is added as laminar flow along the bundle, with the hydraulic diameter set by the packing fraction. Both profiles are coupled to the permeation equations by successive substitution.

**Optimisation.** `optimise` scans the bounded design space with a quasi-random (Sobol) sample, then refines the best point that meets the purity and recovery targets with a constrained local optimiser (SLSQP, then COBYLA).

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

The gap between patterns depends on the case. For the Merkel CO₂-capture case it is large (counter-current reaches a 40.7 % stage cut where cross-flow reaches 28.9 %). For the biogas example below it is only a few per cent of area. Run `sweep_stage_cut` for your own feed rather than relying on a rule of thumb.

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

Then open any notebook from `gasmem/notebooks/` in the Colab file browser. Even simpler: click the **Open in Colab** badge at the top of any notebook on GitHub (or in the [Notebooks](#notebooks) table below); its first cell downloads the toolkit by itself.

**Check the installation:**

```bash
python -m pytest -q        # 100 tests, about two minutes
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

At the same 98 % CH₄ purity, a single cross-flow stage (θ = 0.533) recovers about 76 % of the methane with 2 211 m² and 108 kW. Recycling the second-stage permeate raises recovery to about 94 %, at the cost of about 26 % more area and about 29 % more compression power. `python scripts/multistage_biogas.py` compares the single-, two- and three-stage layouts side by side.

### 6. Screening: which limit am I against?

```python
binary_limits(alpha=35, phi=10, x=0.4)
# {'y_selectivity_limit': 0.959, 'y_pressure_ratio_limit': 1.0, 'y_local': 0.947,
#  'phi_over_alpha': 0.286, 'regime': 'pressure-ratio limited'}
```

### 7. Let the optimiser choose the stage cuts and pressure

```python
def design(theta1, theta2, p_high):
    return two_stage_stripping(m, xf, nm3h_to_mols(1000),
                               StageSpec(theta1, p_high, 1, "cross_flow"),
                               StageSpec(theta2, p_high, 1, "cross_flow"))

best = optimise(design, {"theta1": (0.15, 0.6), "theta2": (0.15, 0.7), "p_high": (5, 20)},
                product="slow-gas product", component="CH4",
                purity=0.98, recovery=0.97, objective="cost")
print(best.summary())
```

```
optimum (feasible; 72 flowsheet evaluations): SLSQP: Optimization terminated successfully
  theta1 = 0.2905, theta2 = 0.4013, p_high = 20
  cost = 69.56 USD/t; purity 0.9800, recovery 0.9700
  total area 924.6 m2, total power 192.5 kW, recycle/feed 0.398
```

The objective can also be `"area"`, `"power"`, `"energy"` or your own function. With the placeholder prices the optimum sits at the highest pressure allowed; raise `CostAssumptions(electricity_per_kWh=...)` and it moves to a lower pressure and more membrane (notebook 09).

### 8. A sweep gas on the permeate side

```python
pol = Membrane(MEMBRANE_PRESETS["Polaris CO2/N2 (Merkel 2010)"])
flue = FEED_PRESETS["flue gas, 13% CO2 (Merkel 2010)"]
r = solve_stage(pol, flue, theta=0.12, p_high=1.1, p_low=0.22, pattern="countercurrent",
                sweep_flow=0.05, sweep_composition={"N2": 1.0})      # 5 % of the feed
print(r.summary())
```

```
countercurrent: stage cut = 0.1200, area = 72.41 m2, p_h/p_l = 1.1/0.22 bar, sweep = 0.05 x feed (CO2 0, N2 1)
  comp     feed  retent.    perm.  rec->perm
   CO2   0.1300   0.0580   0.4643     0.6072
    N2   0.8700   0.9420   0.5357     0.0472
```

Flows are per mol/s of feed here (the default `feed_flow=1.0`), so the area is in m² per mol/s. Notebook 10 shows how the recovery, area and permeate purity change with the sweep ratio.

### 9. Check the model against your own data

```python
from gasmem.userdata import template, compare, fit_permeances
template(["CO2", "CH4"], "my_module.csv")               # fill in one row per operating point
c = compare(m, "my_module.csv", pattern="countercurrent")
print(c.summary()); c.plot()                             # bias, RMS, parity plot
fitted, c_fit, info = fit_permeances(m, "my_module.csv") # effective permeances of your module
```

If you do this, please consider [sharing the comparison](CONTRIBUTING.md#share-a-comparison-with-your-data): data from real modules are the most useful contribution to the toolkit.

---

## Notebooks

Each notebook starts from a process question and ends with a design takeaway. They are interactive (ipywidgets) where it helps.

| # | Notebook | Engineering content | Run |
|---|---|---|---|
| 00 | `start_here` | Concepts, the toolkit in five minutes, and which notebook to read next | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SinaGhanbarii/gasmem/blob/main/notebooks/00_start_here.ipynb) |
| 01 | `well_mixed` | The closed-form binary stage, the pressure-ratio and selectivity limits, and a θ slider | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SinaGhanbarii/gasmem/blob/main/notebooks/01_well_mixed.ipynb) |
| 02 | `cross_flow` | The Weller–Steiner check, and a comparison with well-mixed | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SinaGhanbarii/gasmem/blob/main/notebooks/02_cross_flow.ipynb) |
| 03 | `co_and_counter_current` | Shooting the counter-current boundary-value problem, axial profiles, and area savings | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SinaGhanbarii/gasmem/blob/main/notebooks/03_co_and_counter_current.ipynb) |
| 04 | `multicomponent` | Air with water vapour, refinery H₂ purge, and the behaviour of intermediate components | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SinaGhanbarii/gasmem/blob/main/notebooks/04_multicomponent.ipynb) |
| 05 | `module_rating_pressure_drop` | A fixed module, bore pressure drop, turndown, and fibre diameter and length | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SinaGhanbarii/gasmem/blob/main/notebooks/05_module_rating_pressure_drop.ipynb) |
| 06 | `multistage_energy_cost` | Recycle cascades, compressor and vacuum power, and an indicative cost per tonne | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SinaGhanbarii/gasmem/blob/main/notebooks/06_multistage_energy_cost.ipynb) |
| 07 | `literature_validation` | Reproduction of Merkel, Coker, Pan and Feng operating points | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SinaGhanbarii/gasmem/blob/main/notebooks/07_literature_validation.ipynb) |
| 08 | `experimental_data_point_by_point` | Digitised published curves compared point by point, with error statistics | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SinaGhanbarii/gasmem/blob/main/notebooks/08_experimental_data_point_by_point.ipynb) |
| 09 | `design_optimisation` | Stage cut for a specification; optimum stage cuts and pressure for a cascade; price sensitivity | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SinaGhanbarii/gasmem/blob/main/notebooks/09_design_optimisation.ipynb) |
| 10 | `sweep_and_shell_pressure_drop` | Sweep gas in CO₂ capture (with an independent solver check); shell-side pressure drop in a fixed module | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SinaGhanbarii/gasmem/blob/main/notebooks/10_sweep_and_shell_pressure_drop.ipynb) |
| 11 | `your_own_data` | Compare the models with your measurements, fit permeances, and share the results | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SinaGhanbarii/gasmem/blob/main/notebooks/11_your_own_data.ipynb) |
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
| Polaris pure-gas (MTR 2012) | CO₂ capture, full permeance set | Merkel et al., 2012 (MTR / DOE report) |
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
| Counter-current with sweep | Independent collocation solver (`scipy.integrate.solve_bvp`) | – | area and retentate agree to 1 part in 10⁶ |
| Shell-side pressure drop | Analytical limit of negligible permeation | – | agrees to 1 part in 10⁶ (all three patterns) |

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
- **Pressure drop in rating mode only.** Bore-side and (optionally) shell-side pressure drops are modelled when a fixed module is rated; design mode assumes constant pressures. The shell-side estimate treats the bundle as parallel laminar channels; ports, baffles and uneven packing usually add more, which `shell_friction_factor` lets you match to a measured value.
- **Sweep** is available for co-current and counter-current stages in design mode (and in flowsheets), not in rating mode.
- **The optimiser finds a local optimum** from the best point of its scan. For important designs, inspect `history` or repeat with a larger `n_scan` and another `seed`.
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
│   ├── hollowfiber.py      co-current and counter-current (shooting BVP), sweep gas
│   ├── module.py           HollowFibreModule, rating with bore- and shell-side pressure drop
│   ├── stagecut.py         solve_stage, sweep_stage_cut, binary_limits
│   ├── flowsheets.py       single, two- and three-stage cascades with recycle
│   ├── energy.py           compression, vacuum, specific energy and cost
│   ├── optimise.py         stage_cut_for_spec and the flowsheet optimiser
│   ├── userdata.py         compare the models with your data; fit permeances
│   ├── robeson.py          Robeson upper bounds (1991, 2008)
│   ├── properties.py       literature-based membrane and feed presets
│   ├── datasets.py         loader for the digitised validation data
│   └── data/*.csv          digitised data from Pan, Feng and Makaruk
├── notebooks/              00–11 interactive notebooks
├── scripts/                command-line studies (stage-cut sweep, biogas cascades)
├── tests/                  model and literature regression tests
├── tools/                  notebook builder and digitisation records
├── CHANGELOG.md            what changed in each version
└── CONTRIBUTING.md         how to share data, report bugs and contribute code
```

The notebooks are generated from `tools/build_notebooks.py`. To change a notebook, edit the builder and run `python tools/build_notebooks.py --run 03` (or omit the number to rebuild all of them).

---

## Citing, licence, contributing

If `gasmem` helps your work, please cite it:

> Ghanbari, M. S. (2026). *gasmem: stage cut versus performance in gas membrane separators* (v1.2.0). https://github.com/SinaGhanbarii/gasmem

Please also cite the original papers behind any preset or dataset you use.

The code is released under the **MIT licence** (see [LICENSE](LICENSE)). Digitised data points are approximate readings of published figures, provided for model validation. The original figures remain the property of their publishers.

Contributions are welcome. The most useful is a **comparison of the models with your own data**: notebook 11 shows how, and the "Share validation data" issue form collects it. Also welcome:

- validated permeance sets for additional membranes;
- non-isothermal or concentration-polarisation extensions;
- cost correlations with documented sources.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the details, and [CHANGELOG.md](CHANGELOG.md) for what changed in each version.

---

## References

1. Coker, D. T., Freeman, B. D., Fleming, G. K. (1998). Modeling multicomponent gas separation using hollow-fiber membrane contactors. *AIChE Journal* 44, 1289–1302. doi:10.1002/aic.690440607
2. Feng, X., Ivory, J., Rajan, V. S. V. (1999). Air separation by integrally asymmetric hollow-fiber membranes. *AIChE Journal* 45, 2142–2152. doi:10.1002/aic.690451013
3. Hosseini, S. S., et al. (2015). Simulation and sensitivity analysis of transport in asymmetric hollow fiber membrane permeators for air separation. *RSC Advances* 5 (105). doi:10.1039/C5RA13943K
4. Makaruk, A., Miltner, M., Harasek, M. (2010). Membrane biogas upgrading processes for the production of natural gas substitute. *Separation and Purification Technology* 74, 83–92. doi:10.1016/j.seppur.2010.05.010
5. Merkel, T. C., Lin, H., Wei, X., Baker, R. (2010). Power plant post-combustion carbon dioxide capture: an opportunity for membranes. *Journal of Membrane Science* 359, 126–139. doi:10.1016/j.memsci.2009.10.041
6. Merkel, T., et al. (2012). *Membrane Process to Capture CO₂ from Coal-Fired Power Plant Flue Gas.* Final report, US DOE award DE-NT0005312, Membrane Technology and Research, Inc.
7. Pan, C. Y. (1986). Gas separation by high-flux, asymmetric hollow-fiber membrane. *AIChE Journal* 32, 2020–2027. doi:10.1002/aic.690321212
8. Robeson, L. M. (1991). Correlation of separation factor versus permeability for polymeric membranes. *Journal of Membrane Science* 62, 165–185.
9. Robeson, L. M. (2008). The upper bound revisited. *Journal of Membrane Science* 320, 390–400. doi:10.1016/j.memsci.2008.04.030
10. Sidhoum, M., Sengupta, A., Sirkar, K. K. (1988). Asymmetric cellulose acetate hollow fibers: studies in gas permeation. *AIChE Journal* 34, 417–425. doi:10.1002/aic.690340309
11. Weller, S., Steiner, W. A. (1950). Engineering aspects of separation of gases: fractional permeation through membranes. *Chemical Engineering Progress* 46, 585–590.
