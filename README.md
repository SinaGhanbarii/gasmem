# Gas membrane stage-cut toolkit (`gasmem`)

Python models and interactive Jupyter notebooks that show how **stage cut**
(θ = permeate flow / feed flow) controls purity, recovery, membrane area, energy and
cost in gas membrane separators.

## What is included

| Model | Method | Components | Where |
|---|---|---|---|
| Well-mixed (complete mixing) | Closed-form quadratic (binary); single scalar equation (n components) | any | `wellmixed.py` |
| Cross-flow | ODE in retentate flow; Weller–Steiner analytical solution for checking | any (analytical: binary) | `crossflow.py` |
| Co-current plug flow | Initial-value problem | any | `hollowfiber.py` |
| Counter-current plug flow | Boundary-value problem, shooting from the closed end | any | `hollowfiber.py` |
| Fixed module (rating) | Stage cut as output; Hagen–Poiseuille bore pressure drop | any | `module.py` |
| Multistage with recycle | Single, two-stage (enriching / stripping), three-stage; Wegstein-accelerated recycle | any | `flowsheets.py` |
| Energy and cost | Intercooled compression, vacuum pumps, annualised cost | – | `energy.py` |
| Robeson upper bound | 1991 and 2008 bounds, reference polymers, plot | – | `robeson.py` |
| Published presets | Membranes and feeds from Pan, Coker, Feng, Merkel and Makaruk, each with its reference | – | `properties.py` |

Units in the public API: **bar, mol/s, GPU, m², K**.

## Quick start

```bash
pip install -r requirements.txt
jupyter lab notebooks/00_start_here.ipynb
```

```python
from gasmem import *

m = Membrane({"CO2": 100, "CH4": 4})                  # permeances in GPU
r = solve_stage(m, {"CO2": 0.4, "CH4": 0.6}, theta=0.4,
                p_high=15, p_low=1, pattern="countercurrent", feed_flow=1.0)
print(r.summary())                                     # compositions, recovery, area

df = sweep_stage_cut(m, {"CO2": 0.4, "CH4": 0.6}, 15, 1)   # all four patterns, tidy DataFrame

# from permeability (Barrer) and selective-layer thickness (um)
m2 = Membrane.from_permeability({"CO2": 677.8, "CH4": 33.6}, thickness_um=0.5)

# a membrane and feed from a published paper
print(describe("Polaris CO2/N2 (Merkel 2010)"))
polaris = Membrane(MEMBRANE_PRESETS["Polaris CO2/N2 (Merkel 2010)"])
```

## Published presets

| Membrane preset | Paired feed | Source |
|---|---|---|
| Polaris CO2/N2 (1000 / 20 GPU) | flue gas, 13 % CO₂ | Merkel et al. (2010) |
| Polaris pure-gas (CO₂, N₂, CH₄, H₂) | flue gas, 13 % CO₂ | MTR final report to US DOE (2012) |
| Polyimide CO2/CH4, α = 35 (100 GPU CO₂) | biogas, 60 % CH₄ | Makaruk et al. (2010) |
| Cellulose acetate H₂ recovery | ammonia purge gas (H₂, N₂, CH₄, Ar) | Pan (1986) |
| Cellulose acetate CO₂/hydrocarbons | acid gas with C₁–C₃ | Pan (1986) |
| Cellulose acetate H₂S/CO₂ | sour gas, 22 % H₂S | Pan (1986) |
| Polysulfone-like air (N₂, O₂, CO₂, H₂O) | humid air | Coker et al. (1998) |
| Polysulfone-like H₂/hydrocarbons (5 components) | refinery hydrotreater purge | Coker et al. (1998) |
| Asymmetric hollow fibre air | air, 20.5 % O₂ | Feng et al. (1999), via Hosseini et al. (2015) |

`describe(name)` prints the conditions, caveats and full reference for each preset.

## Google Colab

Every notebook finds the toolkit by itself. Use either route:

* **Google Drive (keeps your files):** upload the unzipped `gas-membrane-stagecut-toolkit`
  folder to My Drive, open any notebook from Drive with *Open with → Google Colaboratory*, then
  *Runtime → Run all* and approve Drive access.
* **Quick upload:** in Colab, *File → Upload notebook*, then *Runtime → Run all*. If the
  toolkit is not in your Drive, the first cell asks you to upload
  `gas-membrane-stagecut-toolkit.zip`.

## Notebooks

| Notebook | Content |
|---|---|
| `00_start_here` | Stage cut, the three trade-offs, selectivity- vs pressure-ratio-limited regimes |
| `01_well_mixed` | Closed form, interactive sliders, multicomponent air example |
| `02_cross_flow` | Numerical model against Weller–Steiner, profiles along the module |
| `03_co_and_counter_current` | All four flow patterns, co-current pinch, interactive comparison |
| `04_multicomponent` | Published presets, permeability input, Robeson plot, biogas and a five-component refinery case |
| `05_module_rating_pressure_drop` | Fixed module: feed flow sets the stage cut; bore pressure drop |
| `06_multistage_energy_cost` | Cascades with recycle at Makaruk's biogas conditions, power and cost, stage-cut sweep |
| `07_literature_validation` | The toolkit against Pan (1986), Merkel (2010), Coker (1998), Makaruk (2010) and Feng (1999) |
| `08_experimental_data_point_by_point` | 274 digitised measured points from Pan and Feng, plus Makaruk's model curves, compared point by point |

The notebooks are saved with their outputs, so the static figures render before you run
anything. Sliders need a live kernel with `ipywidgets`. Heavier cells use a **Run Interact**
button.
To regenerate the notebooks from one source file: `python tools/build_notebooks.py --run`.

## Scripts

```bash
python scripts/stage_cut_sweep.py binary --alpha 20 --phi 8 --x-feed 0.2
python scripts/stage_cut_sweep.py preset --membrane "Polaris CO2/N2 (Merkel 2010)" --feed "flue gas, 13% CO2 (Merkel 2010)" --p-high 2 --p-low 0.2
python scripts/stage_cut_sweep.py custom --permeances CO2=1000,N2=20 --feed-comp CO2=0.13,N2=0.87 --p-high 3 --p-low 0.2
python scripts/multistage_biogas.py --pattern countercurrent
```

Each writes a CSV and PNG figures to the current folder.

## Validation against published results (notebook 07)

| Case | Published | Toolkit |
|---|---|---|
| Pan (1986), measured H₂ recovery at θ = 0.55: permeate / residue H₂ | 86.6 % / 8.8 % | 88.6 % / 6.7 % (cross-flow) |
| Merkel et al. (2010) Fig. 6, cross-flow: permeate CO₂, area | 28.9 %, 11.0 million m² | 28.9 %, 11.0 million m² |
| Merkel et al. (2010) Fig. 6, counter-current: permeate CO₂, area | 40.6 %, 6.8 million m² | 40.7 %, 6.4 million m² |
| Merkel et al. (2010) Table 1: area of membranes A and B relative to base | 28 and 2.7 | 30 and 3.0 |
| Coker et al. (1998) refinery module: stage cut | 94.6 % | 94.65 % |
| Feng et al. (1999), measured: N₂ productivity at 98 % vs 95 % purity | about half | 0.57 |
| Makaruk et al. (2010): selectivity giving least membrane area (93.8 % / 81.2 % CH₄ recovery) | about 22 / about 12 | 22 / 12 |
| Makaruk et al. (2010): specific compression power | about 0.3 kWh/Nm³ | 0.24–0.30 kWh/Nm³ |

### Point by point against digitised measurements (notebook 08)

The measured points in the papers' figures were digitised (274 points). Each is stored with its
pixel coordinates in `gasmem/data/`, and `tools/digitised_data.py` documents every axis
calibration. All points are compared with the toolkit at the experiments' conditions:

| Data | Points | Permeances | Toolkit RMS error |
|---|---|---|---|
| Pan (1986) Figs. 2–3: permeate H₂, N₂, CH₄, Ar | 48 | pure gas, no fitting | 0.2–1.5 % (cross-flow) |
| Pan (1986) Figs. 6–7: CO₂, H₂S, C₁–C₃, N₂ | 55 | fitted by Pan to one run | 0.1–1 % |
| Feng et al. (1999) Figs. 4–5: N₂ and O₂, bore-side feed | 44 | pure gas, no fitting | 0.1–0.4 % (matching flow model) |
| Feng et al. (1999) Figs. 13–14: recovery and absolute productivity, 23 and 45 °C | 81 | pure gas + activation energies | O₂ recovery 1.3–1.9 points; N₂ recovery 1–1.8 points below 97.5 % N₂ (4–10 points above); productivity within 0.001–0.007 Nm³/(m² h) |
| Makaruk et al. (2010) Figs. 15–16 (their model curves) | 18 | as in the paper | membrane area 3.4 %, specific power 7 % (RMS difference) |

Limits found this way: the toolkit over-predicts N₂ recovery above about 98 % purity, as Feng's
own model does (concentration polarisation, not modelled). Feng's shell-side-feed runs fall about
0.4 points short of the ideal model, which is not explained by the bore pressure drop.

```python
from gasmem.datasets import load, available
available()                 # every digitised series
pan = load("Pan1986")       # columns: figure, quantity, series, x, y, px, py
```

Where the published values come from models with the same assumptions, agreement is within a
few per cent. Against measurements, the ideal models reproduce trends and land within a few
points of composition, provided the right flow pattern is used (cross-flow for Pan's high-flux
asymmetric fibres). Notebook 07 explains every remaining difference.

## Verification (`pytest tests/`)

75 tests, about 2 minutes in total. `tests/test_literature.py` pins the agreements above
(including point-by-point RMS limits against the digitised data), and
`tests/test_models.py` checks the following:

* the well-mixed closed form equals the general n-component solver;
* numerical cross-flow equals the **Weller–Steiner** analytical solution (to about 1e-9);
* counter-current agrees with an **independent discretised model** (within the discretisation error);
* as θ → 0, all patterns approach the local permeate composition;
* α = 1 gives no separation;
* component mass balances close (better than 1e-8);
* permeate purity ranks counter-current ≥ cross-flow ≥ co-current ≥ well-mixed;
* rating mode inverts design mode;
* bore pressure drop lowers the stage cut monotonically and vanishes for wide bores;
* recycle flowsheets converge and close their balances.

## Assumptions and limits

* Isothermal, ideal gas; constant permeances (no plasticisation, competitive sorption or
  temperature dependence); no concentration polarisation.
* Sweepless modules. Feed-side (shell) pressure drop is neglected; bore-side pressure drop is
  modelled in rating mode for co- and counter-current flow, with the permeate in the bores.
* Presets reproduce the values reported in each paper (often pure-gas values at one
  temperature). Real mixed-gas performance can be lower; for example, MTR report a mixed-gas
  CO₂/N₂ selectivity of 20–25 for Polaris at 50 °C, against 50 for pure gases.
* Cost figures (`CostAssumptions`) are **placeholders** that show how the trade-off moves with
  stage cut. Replace them with your own data before drawing economic conclusions.
* Counter-current solves take about 0.1–1 s; very selective membranes at high stage cut
  (fast gas stripped to ~1e-30) take longer.

## References

Sources of the presets and validation cases (also in `gasmem.LITERATURE`):

* Coker, D.T., Freeman, B.D. and Fleming, G.K. (1998) 'Modeling multicomponent gas separation
  using hollow-fiber membrane contactors', *AIChE Journal*, 44(6), pp. 1289–1302.
  doi:10.1002/aic.690440607
* Feng, X., Ivory, J. and Rajan, V.S.V. (1999) 'Air separation by integrally asymmetric
  hollow-fiber membranes', *AIChE Journal*, 45(10), pp. 2142–2152. doi:10.1002/aic.690451013
* Hosseini, S.S. et al. (2015) 'Simulation and sensitivity analysis of transport in asymmetric
  hollow fiber membrane permeators for air separation', *RSC Advances*, 5. doi:10.1039/C5RA13943K
* Makaruk, A., Miltner, M. and Harasek, M. (2010) 'Membrane biogas upgrading processes for the
  production of natural gas substitute', *Separation and Purification Technology*, 74, pp. 83–92.
  doi:10.1016/j.seppur.2010.05.010
* Merkel, T.C., Lin, H., Wei, X. and Baker, R. (2010) 'Power plant post-combustion carbon dioxide
  capture: an opportunity for membranes', *Journal of Membrane Science*, 359, pp. 126–139.
  doi:10.1016/j.memsci.2009.10.041
* Merkel, T. et al. (2012) *Membrane Process to Capture CO2 from Coal-Fired Power Plant Flue
  Gas*, Final Report, US DOE award DE-NT0005312. Membrane Technology and Research, Inc.
* Pan, C.Y. (1986) 'Gas separation by high-flux, asymmetric hollow-fiber membrane',
  *AIChE Journal*, 32(12), pp. 2020–2027. doi:10.1002/aic.690321212
* Robeson, L.M. (2008) 'The upper bound revisited', *Journal of Membrane Science*, 320,
  pp. 390–400. doi:10.1016/j.memsci.2008.04.030
* Sidhoum, M., Sengupta, A. and Sirkar, K.K. (1988) 'Asymmetric cellulose acetate hollow fibers:
  studies in gas permeation', *AIChE Journal*, 34(3), pp. 417–425. doi:10.1002/aic.690340309
  (sweep-gas data; reserved for a future sweep option)

Background:

* Weller, S. and Steiner, W.A. (1950) 'Engineering aspects of separation of gases: fractional
  permeation through membranes', *Chemical Engineering Progress*, 46, pp. 585–590.
* Geankoplis, C.J. *Transport Processes and Separation Process Principles*, chapter on gas
  permeation membrane processes (complete-mixing and cross-flow models).
* Baker, R.W. *Membrane Technology and Applications*, chapter on gas separation
  (pressure-ratio and selectivity limits, multistage design).
