"""Regenerate the notebooks in ../notebooks from this single source.

    python tools/build_notebooks.py           # write notebooks
    python tools/build_notebooks.py --run     # write and execute them
    python tools/build_notebooks.py --run 08  # only notebook 08
"""
import sys
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks"

SETUP = r"""\
# --- Find the toolkit (works on your own computer and in Google Colab) ---
import sys, os, glob, pathlib

def _find_toolkit():
    here = pathlib.Path.cwd()
    for p in [here, *here.parents]:                       # running from inside the folder
        if (p / "gasmem" / "__init__.py").exists():
            return p
    for pat in ["/content/*/gasmem/__init__.py",           # Colab: repository cloned into /content
                "/content/drive/MyDrive/gasmem/__init__.py",
                "/content/drive/MyDrive/*/gasmem/__init__.py",
                "/content/drive/MyDrive/*/*/gasmem/__init__.py",
                "/content/drive/MyDrive/*/*/*/gasmem/__init__.py"]:
        hits = sorted(glob.glob(pat))
        if hits:
            return pathlib.Path(hits[0]).parents[1]
    return None

TOOLKIT = _find_toolkit()
try:
    import google.colab  # noqa: F401
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

REPO = "https://github.com/SinaGhanbarii/gasmem.git"
if TOOLKIT is None and IN_COLAB:
    # opened from the "Open in Colab" badge: fetch the toolkit from GitHub
    import subprocess
    print("Downloading the toolkit from", REPO)
    if subprocess.run(["git", "clone", "-q", "--depth", "1", REPO, "/content/gasmem"]).returncode:
        print("Download failed; looking in your Google Drive instead (approve the access request)")
        from google.colab import drive
        drive.mount("/content/drive")
    TOOLKIT = _find_toolkit()
if IN_COLAB:
    from google.colab import output
    output.enable_custom_widget_manager()               # makes the sliders work in Colab
if TOOLKIT is None:
    raise FileNotFoundError("Could not find the 'gasmem' folder. Open this notebook from inside "
                            "your copy of the repository (git clone " + REPO + ").")
sys.path.insert(0, str(TOOLKIT))
os.chdir(TOOLKIT / "notebooks")
print("Using toolkit at", TOOLKIT)
# ---------------------------------------------------------------------------

import numpy as np, pandas as pd, matplotlib.pyplot as plt
import ipywidgets as w
from IPython.display import display, Markdown
from gasmem import *
import gasmem.plotting as gasmem_plot
from gasmem.plotting import (plot_stage_cut_tradeoff, plot_purity_recovery, plot_profiles,
                             bar_breakdown, COLORS, MARKERS)
pd.set_option("display.precision", 4)"""


BADGE = ("[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]"
         "(https://colab.research.google.com/github/SinaGhanbarii/gasmem/blob/main/notebooks/{name})")


def md(s):
    return nbf.v4.new_markdown_cell(s.strip("\n"))


def code(s):
    return nbf.v4.new_code_cell(s.strip("\n"))


NOTEBOOKS = {}

# ---------------------------------------------------------------------------
NOTEBOOKS["00_start_here.ipynb"] = [
    md(r"""
# Stage cut versus performance in gas membrane separators

**Stage cut** is the fraction of the feed that passes through the membrane:

$$\theta = \frac{\text{permeate flow}}{\text{feed flow}}$$

It is the main operating knob of a membrane stage. Raising θ always does three things at once:

* **recovery** of the fast gas in the permeate goes **up**,
* **purity** of the permeate goes **down** (more of the slow gas is dragged through),
* the **membrane area** needed goes **up**,

while the retentate becomes purer in the slow gas. How steep these trade-offs are depends on the
membrane selectivity α, the pressure ratio φ = p<sub>feed</sub>/p<sub>permeate</sub>, the feed
composition and the **flow pattern** inside the module.

| Notebook | Model | What it shows |
|---|---|---|
| 01 | Well-mixed (complete mixing) | Closed-form binary solution; the baseline |
| 02 | Cross-flow | Numerical model checked against the Weller–Steiner analytical solution |
| 03 | Co-current and counter-current | Plug flow on both sides; why counter-current wins; pinch behaviour |
| 04 | Multicomponent | Biogas, flue gas, hydrogen and air feeds |
| 05 | Fixed module + pressure drop | Stage cut as an *output*; bore-side pressure drop |
| 06 | Multistage, energy, cost | Recycle cascades; compressor and vacuum power; indicative cost |
| 07 | Validation | Comparison with Pan (1986), Merkel (2010), Coker (1998), Makaruk (2010) and Feng (1999) |
| 08 | Point by point | 274 digitised measurements from Pan and Feng, plus Makaruk's model curves |
| 09 | Design optimisation | Stage cut for a specification; optimum stage cuts and pressures for a cascade |
| 10 | Sweep and shell-side pressure drop | Permeate sweep gas; feed-side pressure drop in a fixed module |
| 11 | Your own data | Compare the models with your measurements, fit permeances, share the results |

Every notebook runs top to bottom. Interactive cells use `ipywidgets`; if the sliders do not
appear, run `pip install ipywidgets` and restart the kernel.
"""),
    code(SETUP),
    md(r"""
## A first look: one binary separation, four flow patterns

Fast gas A at 20 % in the feed, selectivity α = 10, pressure ratio φ = 4.
"""),
    code("""
df = binary_sweep(alpha=10, phi=4, x_feed=0.2, thetas=np.linspace(0.02, 0.95, 30))
fig = plot_stage_cut_tradeoff(df, "A", area_col="area_dimless",
                              area_label="Dimensionless area  A·Q_A·p_h / F",
                              title="α = 10, φ = 4, x_F = 0.20")
"""),
    md(r"""
Two numbers set an upper bound on permeate purity, whatever the stage cut:

* the **selectivity limit**, reached when the pressure ratio is infinite;
* the **pressure-ratio limit**, y ≤ φ·x, reached when the selectivity is infinite.

When φ is much smaller than α the separation is **pressure-ratio limited**: buying a more selective
membrane barely helps. Try it below.
"""),
    code("""
def limits(alpha=10.0, phi=4.0, x=0.2):
    r = binary_limits(alpha, phi, x)
    display(Markdown(
        f"Selectivity limit **{r['y_selectivity_limit']:.3f}** · "
        f"pressure-ratio limit **{r['y_pressure_ratio_limit']:.3f}** · "
        f"actual (θ → 0) **{r['y_local']:.3f}** → *{r['regime']}* (φ/α = {r['phi_over_alpha']:.2f})"))

w.interact(limits,
           alpha=w.FloatLogSlider(value=10, base=10, min=0.1, max=3, step=0.05, description="α"),
           phi=w.FloatLogSlider(value=4, base=10, min=0.05, max=3, step=0.05, description="φ"),
           x=w.FloatSlider(value=0.2, min=0.01, max=0.9, step=0.01, description="x feed"));
"""),
    code("""
# Regime map: best possible permeate purity (theta -> 0) for a lean feed, x_F = 0.1
alphas = np.logspace(0, 3, 70); phis = np.logspace(0.05, 3, 70)
Z = np.array([[binary_limits(a, p, 0.1)["y_local"] for a in alphas] for p in phis])
fig, ax = plt.subplots(figsize=(5.6, 4.4), constrained_layout=True)
cs = ax.contourf(alphas, phis, Z, levels=np.linspace(0.1, 1, 10), cmap="Blues")
ax.contour(alphas, phis, Z, levels=np.linspace(0.1, 1, 10), colors="white", linewidths=0.5)
ax.plot(alphas, alphas, color="#0b0b0b", lw=1.2, ls="--")
ax.text(250, 120, "φ = α", color="white", rotation=37, ha="left", va="top")
ax.text(300, 1.6, "pressure-ratio\\nlimited", color="#0b0b0b", ha="center", fontsize=9)
ax.text(3, 300, "selectivity\\nlimited", color="#0b0b0b", ha="center", fontsize=9)
ax.set_xscale("log"); ax.set_yscale("log"); ax.grid(False)
ax.set_xlabel("Selectivity α"); ax.set_ylabel("Pressure ratio φ")
ax.set_title("Best permeate purity (θ → 0), x_F = 0.10")
fig.colorbar(cs, ax=ax, label="Permeate mole fraction of fast gas");
"""),
    md(r"""
Read the contours. Where α > φ (bottom right) they run **horizontally**: a more selective
membrane does not help, only a higher pressure ratio does. Where φ > α (top left) they run
**vertically**: now selectivity is what counts. Good designs sit near the bend, φ ≈ α.
"""),
]

# ---------------------------------------------------------------------------
NOTEBOOKS["01_well_mixed.ipynb"] = [
    md(r"""
# 1 · Well-mixed (complete mixing) model

Both sides of the membrane are perfectly mixed, so the whole membrane sees the **outlet**
compositions: retentate $x$ and permeate $y$. It is the most pessimistic flow pattern and the only
one with a closed-form solution.

For a binary with selectivity $\alpha = Q_A/Q_B$ and pressure ratio $\varphi = p_h/p_l$:

$$\frac{y}{1-y} = \frac{\alpha\,(x - y/\varphi)}{(1-x) - (1-y)/\varphi}
\qquad\text{(rate equation)}$$

$$x_F = \theta\,y + (1-\theta)\,x \qquad\text{(mass balance)}$$

Substituting $x$ from the mass balance gives a quadratic $a y^2 + b y + c = 0$, so $y$, $x$ and the
area $A = \theta F / J_{tot}$ follow directly.

For **n components** the same two equations combine into
$y_i = \dfrac{Q_i p_h x_{F,i}/(1-\theta)}{J_{tot} + Q_i p_l + Q_i p_h\,\theta/(1-\theta)}$,
and $\sum y_i = 1$ is one equation in the single unknown $J_{tot}$.
"""),
    code(SETUP),
    md("## The closed form, step by step"),
    code("""
sol = well_mixed_binary(alpha=10, phi=4, x_feed=0.209, theta=0.2)
a, b, c = sol["coefficients"]
print(f"quadratic: {a:.4f} y^2 + {b:.4f} y + {c:.4f} = 0")
print(f"permeate y = {sol['y']:.4f}, retentate x = {sol['x']:.4f}, "
      f"dimensionless area = {sol['area_dimless']:.4f}")
"""),
    md("## How selectivity changes the trade-off (φ = 10, x_F = 0.2)"),
    code("""
th = np.linspace(0.01, 0.97, 60)
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
for k, alpha in enumerate([5, 20, 80]):
    d = pd.DataFrame([well_mixed_binary(alpha, 10, 0.2, t) for t in th]).assign(theta=th)
    col = ["#2a78d6", "#eb6834", "#1baf7a"][k]
    ax[0].plot(d["theta"], d["y"], color=col, label=f"α = {alpha}")
    ax[1].plot(d["theta"] * d["y"] / 0.2, d["y"], color=col, label=f"α = {alpha}")
ax[0].set(xlabel="Stage cut θ", ylabel="Permeate purity", title="Purity falls with stage cut", xlim=(0, 1))
ax[1].set(xlabel="Fast-gas recovery", ylabel="Permeate purity", title="Purity–recovery trade-off", xlim=(0, 1))
ax[0].legend(); ax[1].legend();
"""),
    md(r"""
Going from α = 20 to α = 80 gains far less than going from 5 to 20: with φ = 10 the separation
becomes pressure-ratio limited.
"""),
    md("## Interactive: move the stage cut and watch the trade-off"),
    code("""
def explore(alpha=10.0, phi=4.0, x_feed=0.2, theta=0.3):
    th = np.linspace(0.01, 0.97, 60)
    rows = [dict(theta=t, **{k: v for k, v in well_mixed_binary(alpha, phi, x_feed, t).items()
                              if k != "coefficients"}) for t in th]
    d = pd.DataFrame(rows)
    d["recovery"] = d["theta"] * d["y"] / x_feed
    now = well_mixed_binary(alpha, phi, x_feed, theta)
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.3), constrained_layout=True)
    for a_, col, ttl, yl in [(ax[0], "y", "Permeate purity (fast gas)", "Mole fraction"),
                              (ax[1], "recovery", "Fast-gas recovery", "Fraction of feed"),
                              (ax[2], "area_dimless", "Membrane area", "A·Q_A·p_h / F")]:
        a_.plot(d["theta"], d[col], color=COLORS["well_mixed"])
        a_.axvline(theta, color="#52514e", ls="--", lw=1)
        a_.set_title(ttl); a_.set_ylabel(yl); a_.set_xlabel("Stage cut θ"); a_.set_xlim(0, 1)
    ax[0].plot(d["theta"], d["x"], color=COLORS["well_mixed"], ls=":", label="retentate x")
    ax[0].legend()
    plt.show()
    display(Markdown(f"At θ = {theta:.2f}: permeate **{now['y']:.3f}**, retentate **{now['x']:.3f}**, "
                     f"recovery **{theta*now['y']/x_feed:.3f}**, area **{now['area_dimless']:.3f}**"))

w.interact(explore,
           alpha=w.FloatSlider(value=10, min=1.5, max=100, step=0.5, continuous_update=False, description="α"),
           phi=w.FloatSlider(value=4, min=1.2, max=100, step=0.2, continuous_update=False, description="φ"),
           x_feed=w.FloatSlider(value=0.2, min=0.02, max=0.9, step=0.01, continuous_update=False, description="x feed"),
           theta=w.FloatSlider(value=0.3, min=0.01, max=0.97, step=0.01, continuous_update=False, description="θ"));
"""),
    md("## Multicomponent example: oxygen enrichment from air"),
    code("""
m = Membrane(MEMBRANE_PRESETS["Asymmetric hollow fibre air (Feng 1999)"])
print(m)
for th in (0.1, 0.3, 0.6):
    print(well_mixed(m, FEED_PRESETS["air, 20.5% O2 (Feng 1999)"], th, p_high=8.0, p_low=1.0, feed_flow=1.0).summary(), "\\n")
"""),
    md(r"""
**Takeaway.** In a well-mixed stage the entire membrane works against the leanest (retentate)
composition, so permeate purity falls steeply with stage cut. The other flow patterns in
notebooks 02–03 do better because the feed-side composition varies along the module.
"""),
]

# ---------------------------------------------------------------------------
NOTEBOOKS["02_cross_flow.ipynb"] = [
    md(r"""
# 2 · Cross-flow model

The feed moves in plug flow; the permeate leaves each element straight away, without mixing
along the module. Each element therefore releases permeate of the **local** composition
$y(x)$, the solution of $y_i = J_i / \sum J$ with $J_i = Q_i (p_h x_i - p_l y_i)$.

Using the retentate flow $L$ as the independent variable makes the stage cut an input:

$$\frac{dN_i}{dL} = y_i(x), \qquad \frac{dA}{dL} = -\frac{1}{J_{tot}}, \qquad L: F \rightarrow (1-\theta)F$$

**Weller & Steiner (1950)** integrated the binary case analytically. The closed form is coded in
`weller_steiner()` and used here as an independent check of the numerical model.
"""),
    code(SETUP),
    md("## Numerical model versus Weller–Steiner"),
    code("""
alpha, phi, xf = 10, 4, 0.209
rows = []
for th in (0.05, 0.2, 0.4, 0.6, 0.8, 0.95):
    xo, yp = weller_steiner_theta(alpha, phi, xf, th)
    r = cross_flow(Membrane.binary(alpha), [xf, 1 - xf], th, phi, 1.0, profile=False)
    rows.append(dict(theta=th, y_analytical=yp, y_numerical=r.y_perm[0],
                     x_analytical=xo, x_numerical=r.x_ret[0],
                     max_abs_difference=max(abs(yp - r.y_perm[0]), abs(xo - r.x_ret[0]))))
pd.DataFrame(rows).style.format({"max_abs_difference": "{:.1e}"}, precision=5)
"""),
    code("""
th = np.linspace(0.01, 0.97, 50)
fig, ax = plt.subplots(figsize=(5.4, 3.8), constrained_layout=True)
ax.plot(th, [weller_steiner_theta(alpha, phi, xf, t)[1] for t in th], color=COLORS["cross_flow"],
        label="Weller–Steiner (analytical)")
ths = np.linspace(0.05, 0.95, 10)
ax.plot(ths, [cross_flow(Membrane.binary(alpha), [xf, 1 - xf], t, phi, 1.0, profile=False).y_perm[0]
              for t in ths], ls="none", marker="o", color="#0b0b0b", ms=5, label="numerical model")
ax.plot(th, [well_mixed_binary(alpha, phi, xf, t)["y"] for t in th], color=COLORS["well_mixed"],
        label="well-mixed (for reference)")
ax.set(xlabel="Stage cut θ", ylabel="Permeate purity (fast gas)", xlim=(0, 1),
       title="Cross-flow: α = 10, φ = 4, x_F = 0.209"); ax.legend();
"""),
    md("## Interactive: cross-flow against the well-mixed baseline"),
    code("""
def explore(alpha=10.0, phi=4.0, x_feed=0.2, theta=0.4):
    df = binary_sweep(alpha, phi, x_feed, np.linspace(0.02, 0.95, 28), ["well_mixed", "cross_flow"])
    plot_stage_cut_tradeoff(df, "A", "area_dimless", ["well_mixed", "cross_flow"], theta_now=theta,
                            area_label="A·Q_A·p_h / F"); plt.show()
    r = cross_flow(Membrane.binary(alpha), [x_feed, 1 - x_feed], theta, phi, 1.0)
    plot_profiles({"cross_flow": r}, "A", title="Along the module: feed side and local permeate"); plt.show()

w.interact(explore,
           alpha=w.FloatSlider(value=10, min=1.5, max=100, step=0.5, continuous_update=False, description="α"),
           phi=w.FloatSlider(value=4, min=1.2, max=100, step=0.2, continuous_update=False, description="φ"),
           x_feed=w.FloatSlider(value=0.2, min=0.02, max=0.9, step=0.01, continuous_update=False, description="x feed"),
           theta=w.FloatSlider(value=0.4, min=0.02, max=0.95, step=0.01, continuous_update=False, description="θ"));
"""),
    md(r"""
**Takeaway.** Near the feed inlet the cross-flow module produces permeate much richer than the
average, so for the same stage cut the collected permeate is purer, and slightly less area is
needed, than in a well-mixed stage.
"""),
]

# ---------------------------------------------------------------------------
NOTEBOOKS["03_co_and_counter_current.ipynb"] = [
    md(r"""
# 3 · Co-current and counter-current plug flow

Both sides are in plug flow (a sweepless hollow-fibre module). The permeate starts with zero flow
at its closed end, where its composition equals the local cross-flow value.

* **Co-current**: the permeate flows with the feed, an initial-value problem in $L$.
* **Counter-current**: the permeate flows against the feed and leaves at the feed end, a
  two-point boundary-value problem. It is solved by **shooting from the closed (retentate) end**:
  guess the retentate composition, integrate back to the feed end, and correct the guess until the
  feed composition is recovered. In rare extreme cases (very selective membranes at high stage
  cut) the fast gas is stripped below double precision; the solver then switches to the
  analytical stripped limit.

Both models are checked in `tests/`: against the cross-flow limit as θ → 0, and against an
independent discretised counter-current model.
"""),
    code(SETUP),
    md("## All four flow patterns over the full range of stage cut"),
    code("""
df = binary_sweep(alpha=20, phi=8, x_feed=0.2, thetas=np.linspace(0.02, 0.95, 28))
plot_stage_cut_tradeoff(df, "A", "area_dimless", area_label="A·Q_A·p_h / F",
                        title="α = 20, φ = 8, x_F = 0.20"); plt.show()
plot_purity_recovery(df, "A"); plt.show()
"""),
    md(r"""
Counter-current is best because the leanest feed (near the retentate end) faces the leanest
permeate, and the richest feed faces the permeate that is about to leave. The local driving force
stays high along the whole module.

In **co-current** flow the bulk permeate is carried along with the feed. Once the feed-side
fast-gas fraction falls to about $y/\varphi$ there is no driving force left for the fast gas, and
the feed cannot be stripped further. This **pinch** is visible in the retentate below.
"""),
    code("""
m = Membrane.binary(20)
res = {p: solve_stage(m, [0.2, 0.8], 0.6, 8.0, 1.0, p, profile=True)
       for p in ["cross_flow", "cocurrent", "countercurrent"]}
plot_profiles(res, "A", title="θ = 0.6: fast-gas fraction along the module"); plt.show()
pd.DataFrame({LABELS[p]: {"permeate y_A": r.y_perm[0], "retentate x_A": r.x_ret[0],
                          "area (m2, F = 1 mol/s)": r.area} for p, r in res.items()}).T
"""),
    md("## Interactive comparison"),
    code("""
def explore(alpha=20.0, phi=8.0, x_feed=0.2, theta=0.5):
    df = binary_sweep(alpha, phi, x_feed, np.linspace(0.03, 0.93, 20))
    plot_stage_cut_tradeoff(df, "A", "area_dimless", theta_now=theta, area_label="A·Q_A·p_h / F")
    plt.show()
    pts = df.loc[(df["theta"] - theta).abs().groupby(df["pattern"]).idxmin()]
    display(pts[["pattern", "theta", "purity", "recovery", "retentate_slow_purity", "area_dimless"]]
            .set_index("pattern"))

w.interact(explore,
           alpha=w.FloatSlider(value=20, min=1.5, max=100, step=0.5, continuous_update=False, description="α"),
           phi=w.FloatSlider(value=8, min=1.2, max=100, step=0.2, continuous_update=False, description="φ"),
           x_feed=w.FloatSlider(value=0.2, min=0.02, max=0.9, step=0.01, continuous_update=False, description="x feed"),
           theta=w.FloatSlider(value=0.5, min=0.03, max=0.93, step=0.01, continuous_update=False, description="θ"));
"""),
    md(r"""
**Things to try**

* Set φ ≫ α (selectivity limited): the four patterns move closer together.
* Set φ ≪ α (pressure-ratio limited): flow pattern matters most, and counter-current pulls ahead.
* Move θ towards 1: every pattern converges to the feed composition in the permeate, but the
  retentate purity differs a lot.
"""),
]

# ---------------------------------------------------------------------------
NOTEBOOKS["04_multicomponent.ipynb"] = [
    md(r"""
# 4 · Multicomponent feeds and published membranes

All four models handle any number of components. The membrane and feed presets below are taken
from published papers. `describe()` prints the conditions and the full reference for each one,
and `LITERATURE` holds the reference list.
"""),
    code(SETUP),
    code("""
for k, v in MEMBRANE_PRESETS.items():
    print(f"{k:48s}", {c: round(q, 3) for c, q in v.items()}, "GPU")
print()
print(describe("Cellulose acetate H2 recovery (Pan 1986)"))
"""),
    md(r"""
## Your own membrane from permeability data

Papers usually report **permeability** (Barrer). The modules need **permeance** (GPU), which
depends on the thickness of the selective layer: permeance [GPU] = permeability [Barrer] /
thickness [µm]. `Membrane.from_permeability` does the conversion, and the Robeson plot shows
where the material sits against the best known polymers.
"""),
    code("""
from gasmem.robeson import plot_upper_bound
# 6FDA-durene polyimide (Robeson 2008, Table 2): P(CO2) = 677.8 Barrer, CO2/CH4 = 20.18
mine = Membrane.from_permeability({"CO2": 677.8, "CH4": 677.8 / 20.18}, thickness_um=0.5)
print(mine)
print("fraction of the 2008 upper-bound selectivity:",
      round(distance_to_bound("CO2/CH4", 677.8, 20.18), 2))
plot_upper_bound("CO2/CH4", points={"6FDA-durene": (677.8, 20.18),
                                     "Makaruk 2010 preset (a = 35)": (100 * 0.1, 35)}); plt.show()
"""),
    md(r"""
The Makaruk preset is plotted as if its selective layer were 0.1 µm thick, an assumption made
only to place it on this chart: 100 GPU then corresponds to 10 Barrer.
"""),
    md("## Biogas upgrading at the conditions of Makaruk et al. (2010)"),
    code("""
m = Membrane(MEMBRANE_PRESETS["Polyimide CO2/CH4, a=35 (Makaruk 2010)"])
F = nm3h_to_mols(1000)          # 1000 Nm3/h raw biogas, 10 bar, 60 % CH4
df = sweep_stage_cut(m, FEED_PRESETS["biogas, 60% CH4 (Makaruk 2010)"], p_high=10, p_low=1, feed_flow=F,
                     thetas=np.linspace(0.05, 0.56, 16))
plot_stage_cut_tradeoff(df, "CO2", slow="CH4",
                        title="Biogas, 10 bar / 1 bar, 1000 Nm³/h, α = 35"); plt.show()
"""),
    md(r"""
For biogas the useful product is the **retentate**, which is methane-rich. Its purity climbs with
stage cut, but so does the methane lost to the permeate. This is the classic single-stage
dilemma, and it is why notebook 06 looks at multistage layouts.
"""),
    code("""
d = df.copy()
fig, ax = plt.subplots(figsize=(5.4, 4), constrained_layout=True)
for p in PATTERNS:
    s = d[d["pattern"] == p]
    ax.plot(s["recovery_perm_CH4"], s["x_ret_CH4"], color=COLORS[p], marker=MARKERS[p],
            markevery=3, markersize=4, label=LABELS[p])
ax.axhline(0.98, color="#52514e", lw=1, ls=":"); ax.text(0.01, 0.982, "98 % target (Makaruk 2010)", color="#52514e", fontsize=8)
ax.set_xlabel("CH₄ lost to permeate (fraction of feed CH₄)"); ax.set_ylabel("CH₄ purity of retentate")
ax.set_title("Biomethane purity versus methane loss"); ax.legend();
"""),
    md(r"""
## Five components: hydrogen from a refinery purge (Coker et al. 1998)

With three or more components something new appears. A component of **intermediate**
permeability (here methane) can pass through a concentration maximum inside the module: the
fast gas leaves first and enriches it, then it is stripped in turn. Coker et al. point out that
this could be exploited by staging.
"""),
    code("""
m = Membrane(MEMBRANE_PRESETS["Polysulfone-like H2/hydrocarbons (Coker 1998)"])
xf = FEED_PRESETS["refinery hydrotreater purge (Coker 1998)"]
r = countercurrent(m, xf, 0.85, 42.4, 7.9, nm3h_to_mols(283.2))
fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
for i, c in enumerate(m.components):
    ax.plot(r.profile["area_fraction"], r.profile["x"][i], color=["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"][i], label=c)
ax.set(xlabel="Fraction of membrane area from the feed inlet", ylabel="Feed-side mole fraction",
       title="Refinery purge, counter-current, θ = 0.85", xlim=(0, 1), ylim=(0, None)); ax.legend();
print(r.summary())
"""),
    md("## Interactive: any published preset, any pressures"),
    code("""
def explore(membrane="Polaris CO2/N2 (Merkel 2010)", p_high=2.0, p_low=0.2, theta_max=0.8):
    m = Membrane(MEMBRANE_PRESETS[membrane])
    feed = PAIRINGS[membrane]
    xf = FEED_PRESETS[feed]
    fast = max(m.gpu, key=m.gpu.get)
    slow = max(xf, key=lambda c: xf[c] if c != fast else -1)
    display(Markdown(describe(membrane).replace("\\n", "  \\n")))
    df = sweep_stage_cut(m, xf, p_high, p_low, thetas=np.linspace(0.03, theta_max, 14))
    plot_stage_cut_tradeoff(df, fast, slow=slow, title=f"{membrane}  ·  {feed}"); plt.show()

w.interact_manual(explore,
                  membrane=w.Dropdown(options=list(MEMBRANE_PRESETS), value="Polaris CO2/N2 (Merkel 2010)"),
                  p_high=w.FloatSlider(value=2.0, min=1.0, max=80, step=0.5, description="p feed (bar)"),
                  p_low=w.FloatSlider(value=0.2, min=0.05, max=10, step=0.05, description="p perm (bar)"),
                  theta_max=w.FloatSlider(value=0.8, min=0.2, max=0.95, step=0.05, description="max θ"));
"""),
    md("Press **Run Interact** after changing the inputs. Each preset uses the feed from its own paper. Multicomponent counter-current sweeps take a few seconds."),
]

# ---------------------------------------------------------------------------
NOTEBOOKS["05_module_rating_pressure_drop.ipynb"] = [
    md(r"""
# 5 · A fixed module: stage cut as an output, and bore pressure drop

So far the stage cut was specified and the area calculated (*design* mode). On a real plant the
module already exists, and the stage cut follows from the **feed flow** and the pressures
(*rating* mode). `rate_module()` solves that problem.

**Bore-side pressure drop.** With the feed on the shell side and the permeate inside fibre bores,
the permeate pressure rises towards the closed end of each fibre (Hagen–Poiseuille for a
compressible gas):

$$\frac{dp}{dx} = -\frac{128\,\mu\,R\,T\,q}{\pi\, d_i^4\, n_f\, p}$$

That eats into the pressure ratio exactly where the separation is hardest. Narrow or long fibres
make it worse.

For counter-current flow the solver iterates between the shooting solution and the bore-pressure
profile, which integrates in closed form once the permeate flow profile is known:
$p(z)^2 = p_{out}^2 + 2K\int_0^z \mu\,q\,dz'$.
"""),
    code(SETUP),
    md("## Size a module, then vary the feed flow"),
    code("""
m = Membrane(MEMBRANE_PRESETS["Polyimide CO2/CH4, a=35 (Makaruk 2010)"])
xf = FEED_PRESETS["biogas, 60% CH4 (Makaruk 2010)"]
F_design = nm3h_to_mols(100)
design = countercurrent(m, xf, 0.40, 15, 1, F_design, profile=False)
module = HollowFibreModule.for_area(design.area, length=1.0, d_inner=150e-6, d_outer=300e-6)
print(module)
"""),
    code("""
flows = F_design * np.linspace(0.5, 2.0, 9)
rows = []
for F in flows:
    for dp in (False, True):
        r = rate_module(m, xf, F, 15, 1, module, "countercurrent", pressure_drop=dp, profile=False)
        rows.append(dict(feed_Nm3h=mols_to_nm3h(F), pressure_drop=dp, theta=r.theta,
                         CH4_purity=r.x_ret[1], CH4_recovery=r.recovery("CH4", "retentate"),
                         bore_dp_bar=r.extra["bore_dp_bar"]))
rt = pd.DataFrame(rows)
fig, ax = plt.subplots(1, 3, figsize=(11.5, 3.4), constrained_layout=True)
for dp, ls, lab in [(False, "-", "no pressure drop"), (True, "--", "with bore pressure drop")]:
    s = rt[rt["pressure_drop"] == dp]
    ax[0].plot(s["feed_Nm3h"], s["theta"], color=COLORS["countercurrent"], ls=ls, marker="o", ms=4, label=lab)
    ax[1].plot(s["theta"], s["CH4_purity"], color=COLORS["countercurrent"], ls=ls, marker="o", ms=4, label=lab)
    ax[2].plot(s["feed_Nm3h"], s["bore_dp_bar"], color=COLORS["countercurrent"], ls=ls, marker="o", ms=4, label=lab)
ax[0].set(xlabel="Feed flow (Nm³/h)", ylabel="Stage cut θ", title="More feed → lower stage cut")
ax[1].set(xlabel="Stage cut θ", ylabel="CH₄ purity of retentate", title="Same module, varying feed")
ax[2].set(xlabel="Feed flow (Nm³/h)", ylabel="Bore pressure drop (bar)", title="Pressure drop")
ax[0].legend(); plt.show()
rt.round(4)
"""),
    md("## Interactive: fibre bore diameter and length"),
    code("""
def explore(d_inner_um=150, length_m=1.0, feed_Nm3h=100.0, pattern="countercurrent"):
    mod = HollowFibreModule.for_area(design.area, length=length_m, d_inner=d_inner_um * 1e-6,
                                     d_outer=2 * d_inner_um * 1e-6)
    F = nm3h_to_mols(feed_Nm3h)
    ideal = rate_module(m, xf, F, 15, 1, mod, pattern, pressure_drop=False)
    real = rate_module(m, xf, F, 15, 1, mod, pattern, pressure_drop=True)
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.4), constrained_layout=True)
    ax[0].plot(real.profile["area_fraction"], real.profile["p_perm_bar"], color=COLORS[pattern])
    ax[0].set(xlabel="Fraction of fibre length from the feed inlet", ylabel="Bore pressure (bar)",
              title="Permeate pressure along the fibre"); ax[0].set_ylim(bottom=0)
    i = m.components.index("CO2")
    for r, ls, lab in [(ideal, "-", "ideal"), (real, "--", "with pressure drop")]:
        ax[1].plot(r.profile["area_fraction"], r.profile["x"][i], color=COLORS[pattern], ls=ls, label=lab)
    ax[1].set(xlabel="Fraction of fibre length from the feed inlet", ylabel="CO₂ in feed side",
              title="Feed-side CO₂"); ax[1].legend(); plt.show()
    display(pd.DataFrame({"ideal": [ideal.theta, ideal.x_ret[1], 0.0],
                          "with pressure drop": [real.theta, real.x_ret[1], real.extra["bore_dp_bar"]]},
                         index=["stage cut", "CH4 purity (retentate)", "bore Δp (bar)"]).round(4))
    print(mod)

w.interact_manual(explore,
                  d_inner_um=w.IntSlider(value=150, min=60, max=600, step=10, description="bore (µm)"),
                  length_m=w.FloatSlider(value=1.0, min=0.3, max=3.0, step=0.1, description="length (m)"),
                  feed_Nm3h=w.FloatSlider(value=100, min=40, max=250, step=5, description="feed Nm³/h"),
                  pattern=w.Dropdown(options=["countercurrent", "cocurrent"]));
"""),
    md(r"""
**Takeaway.** Stage cut is not set directly on a plant: it follows from feed flow, pressures and
module geometry. Bore pressure drop lowers the effective pressure ratio, which reduces both the
stage cut and the separation, most of all for narrow, long fibres.
"""),
]

# ---------------------------------------------------------------------------
NOTEBOOKS["06_multistage_energy_cost.ipynb"] = [
    md(r"""
# 6 · Multistage cascades, energy and cost

A single stage cannot give high purity *and* high recovery at the same time. Cascades with recycle
work around this at the price of more area and more compression. This notebook uses biogas
upgrading at the conditions of Makaruk et al. (2010): 1000 Nm³/h of raw biogas with 60 % CH₄,
compressed to 10 bar, a polyimide membrane with CO₂ permeance 100 GPU and CO₂/CH₄ = 35, and a
target of 98 % CH₄ in the product.

Layouts (each stage can use any flow pattern):

* **Single stage**
* **Two-stage stripping**: retentate 1 is polished in stage 2, and permeate 2 is recompressed and recycled.
* **Two-stage enriching**: permeate 1 is recompressed and purified in stage 2, and retentate 2 is recycled.
* **Three-stage**: retentate 1 → stage 2 (biomethane); permeate 1 → stage 3 at the stage-1
  permeate pressure (CO₂ off-gas); permeate 2 and retentate 3 are recycled to the compressor.

Recycles are converged by Wegstein-accelerated successive substitution.

**Energy** is computed for intercooled two-stage compression with Makaruk's efficiencies
(isentropic 0.80 × mechanical 0.90 = 0.72).
**Cost** uses the transparent placeholder values in `CostAssumptions`. They show how the trade-off
*moves* with stage cut, not what a plant would cost; replace them with your own figures.
"""),
    code(SETUP),
    code("""
m = Membrane(MEMBRANE_PRESETS["Polyimide CO2/CH4, a=35 (Makaruk 2010)"])
xf = FEED_PRESETS["biogas, 60% CH4 (Makaruk 2010)"]
F = nm3h_to_mols(1000)
ea = EnergyAssumptions(eta_compressor=0.8 * 0.9)
ca = CostAssumptions()
pattern = "cross_flow"      # fast; use "countercurrent" for hollow fibres (slower)
pd.Series(vars(ca), name="value").to_frame()
"""),
    md("## Four layouts side by side"),
    code("""
S = lambda th, ph, pl: StageSpec(th, ph, pl, pattern)
cases = {
    "single stage": single_stage(m, xf, F, S(0.533, 10, 1), ea=ea),
    "two-stage stripping": two_stage_stripping(m, xf, F, S(0.330, 10, 1), S(0.335, 10, 1), ea=ea),
    "two-stage enriching": two_stage_enriching(m, xf, F, S(0.50, 10, 1), S(0.70, 10, 1), ea=ea),
    "three-stage": three_stage(m, xf, F, S(0.40, 10, 2), S(0.30, 10, 1), S(0.70, 2, 1), ea=ea),
}
rows = {}
for name, r in cases.items():
    s = r.products["slow-gas product"]
    rows[name] = {"CH4 purity": r.purity(s, "CH4"), "CH4 recovery": r.recovery(s, "CH4"),
                  "recycle / feed": r.recycle_ratio(), "area (m2)": r.total_area,
                  "power (kW)": r.total_power_kW,
                  "kWh per Nm3 product": r.total_power_kW / mols_to_nm3h(r.streams[s].sum()),
                  "kWh per t CH4": specific_energy_kWh_per_tonne(r, s, "CH4"),
                  "USD per t CH4 (indicative)": specific_cost_per_tonne(r, s, "CH4", ca)}
summary = pd.DataFrame(rows).T
summary.round(3)
"""),
    code("""
print(cases["three-stage"].summary())
cases["three-stage"].stream_table().round(4)
"""),
    code("""
r = cases["three-stage"]
bar_breakdown(list(r.power_kW), list(r.power_kW.values()), "kW", "Three-stage: power by machine"); plt.show()
c = {k.replace("_USD_per_yr", "").replace("_", " "): v for k, v in r.cost.items() if k.endswith("per_yr") and not k.startswith("total")}
bar_breakdown(list(c), list(c.values()), "USD per year (indicative)", "Three-stage: annualised cost"); plt.show()
"""),
    md(r"""
**Reading the table.** Every layout here was tuned to give about 98 % CH₄.
- **Single stage:** reaches that purity only by losing about a quarter of the methane to the permeate.
- **Two-stage stripping:** recycling the second permeate raises methane recovery to about 94 %. This is the layout Makaruk et al. recommend.
- **Two-stage enriching:** reaches about 96 % recovery. The second compressor makes its power higher, but it also sends a purer CO₂ stream out of the plant.
- **Three-stage:** recovers about 98 %, but needs much more membrane area, because stage 3 runs at a low pressure ratio of 2.

The specific power of about 0.25–0.3 kWh per Nm³ of product is close to the 0.3 kWh/Nm³
Makaruk et al. report. Notebook 07 checks this case in detail.
"""),
    md(r"""
## Stage cut of the first stage: where is the optimum?

The stage-2 and stage-3 stage cuts are held fixed while the stage-1 stage cut is swept.
"""),
    code("""
rows = []
for th1 in np.linspace(0.3, 0.5, 11):
    r = three_stage(m, xf, F, S(th1, 10, 2), S(0.30, 10, 1), S(0.70, 2, 1), ea=ea)
    rows.append(dict(theta1=th1, purity=r.purity("retentate 2", "CH4"),
                     recovery=r.recovery("retentate 2", "CH4"), area=r.total_area,
                     power=r.total_power_kW, cost=specific_cost_per_tonne(r, "retentate 2", "CH4", ca)))
sw = pd.DataFrame(rows)
fig, ax = plt.subplots(1, 4, figsize=(13, 3.2), constrained_layout=True)
for a, col, ttl in zip(ax, ["purity", "recovery", "power", "cost"],
                       ["CH₄ purity (product)", "CH₄ recovery", "Total power (kW)", "USD / t CH₄ (indicative)"]):
    a.plot(sw["theta1"], sw[col], color=COLORS[pattern], marker="o", ms=4)
    a.set_title(ttl); a.set_xlabel("Stage-1 stage cut θ₁")
plt.show()
sw.round(4)
"""),
    md("## Interactive three-stage design"),
    code("""
def explore(theta1=0.40, theta2=0.30, theta3=0.7, p_feed=10.0, p_mid=2.0, pattern="cross_flow"):
    r = three_stage(m, xf, F, StageSpec(theta1, p_feed, p_mid, pattern),
                    StageSpec(theta2, p_feed, 1.0, pattern), StageSpec(theta3, p_mid, 1.0, pattern), ea=ea)
    display(Markdown(
        f"**Biomethane**: CH₄ {r.purity('retentate 2','CH4'):.4f}, recovery {r.recovery('retentate 2','CH4'):.4f}  \\n"
        f"**Off-gas**: CO₂ {r.purity('permeate 3','CO2'):.4f}, CH₄ slip {1-r.recovery('retentate 2','CH4'):.4f} of feed CH₄  \\n"
        f"Area {r.total_area:,.0f} m² · power {r.total_power_kW:,.1f} kW · recycle/feed {r.recycle_ratio():.3f} · "
        f"{specific_energy_kWh_per_tonne(r,'retentate 2','CH4'):,.0f} kWh/t CH₄ · "
        f"{specific_cost_per_tonne(r,'retentate 2','CH4', ca):,.0f} USD/t CH₄ (indicative)"))
    display(r.stream_table().round(4))

w.interact_manual(explore,
                  theta1=w.FloatSlider(value=0.40, min=0.15, max=0.6, step=0.01, description="θ₁"),
                  theta2=w.FloatSlider(value=0.30, min=0.03, max=0.5, step=0.01, description="θ₂"),
                  theta3=w.FloatSlider(value=0.70, min=0.3, max=0.95, step=0.01, description="θ₃"),
                  p_feed=w.FloatSlider(value=10, min=4, max=30, step=0.5, description="p feed"),
                  p_mid=w.FloatSlider(value=2, min=1.25, max=8, step=0.25, description="p perm 1"),
                  pattern=w.Dropdown(options=["cross_flow", "countercurrent", "cocurrent", "well_mixed"]));
"""),
]

# ---------------------------------------------------------------------------
NOTEBOOKS["07_literature_validation.ipynb"] = [
    md(r"""
# 7 · Validation against published results

This notebook reruns cases from five papers and compares the toolkit's numbers with the published
ones. Some published values are **measurements**; others are **results from the authors' own
models**. Both are useful checks, but only measurements test the physics, so each case says which
kind it is.

| Paper | Kind of data | What is compared |
|---|---|---|
| Pan (1986) | measurement | H₂ recovery from ammonia purge gas, cellulose acetate hollow fibres |
| Merkel et al. (2010) | model results | flue-gas CO₂ capture: pressure-ratio limit, cross-flow against counter-current, three membranes |
| Coker et al. (1998) | model results | a five-component refinery H₂ module with bore pressure drop |
| Makaruk et al. (2010) | model results | two-stage biogas cascade: specific power and membrane area |
| Feng et al. (1999) | measurement (rules stated in the text) | air separation: purity–productivity, temperature and pressure effects |

The full references are in `LITERATURE`.
"""),
    code(SETUP),
    code("""
checks = []          # collected at the end
def check(case, quantity, paper, model, kind):
    checks.append(dict(case=case, quantity=quantity, published=paper, toolkit=model, data=kind))
for k in ("Pan1986", "Merkel2010", "Coker1998", "Makaruk2010", "Feng1999"):
    print("-", LITERATURE[k])
"""),
    md(r"""
## 1 · Pan (1986): hydrogen recovery, measured

Pan ran a small module of asymmetric cellulose acetate hollow fibres on a simulated ammonia purge
gas: 51.78 % H₂, 24.69 % N₂, 19.57 % CH₄ and 3.96 % Ar, at 6964 kPa feed, 1123 kPa permeate and
25 °C. The pure-gas permeabilities are in the preset.

The text gives one measured point in full: at a stage cut of 0.55, run co-currently, the permeate
contained **86.6 % H₂** and the residue **8.8 % H₂**.

Pan's main conclusion was that in high-flux *asymmetric* membranes the porous support stops local
permeate from mixing. The skin therefore sees **cross-flow** whatever the module's flow
arrangement.
"""),
    code("""
m = Membrane(MEMBRANE_PRESETS["Cellulose acetate H2 recovery (Pan 1986)"])
xf = FEED_PRESETS["ammonia purge gas, NH3-free (Pan 1986)"]
rows = {}
for p in PATTERNS:
    r = solve_stage(m, xf, 0.55, 69.64, 11.23, p)
    rows[LABELS[p]] = {"permeate H2": r.y_perm[0], "residue H2": r.x_ret[0]}
rows["Measured (Pan, co-current run)"] = {"permeate H2": 0.866, "residue H2": 0.088}
pan = pd.DataFrame(rows).T
check("Pan 1986, θ = 0.55", "permeate H2", 0.866, pan.loc["Cross-flow", "permeate H2"], "measured")
check("Pan 1986, θ = 0.55", "residue H2", 0.088, pan.loc["Cross-flow", "residue H2"], "measured")
pan.round(3)
"""),
    code("""
th = np.linspace(0.2, 0.75, 12)
fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.6), constrained_layout=True)
for p in ["cross_flow", "cocurrent", "countercurrent"]:
    res = [solve_stage(m, xf, t, 69.64, 11.23, p) for t in th]
    ax[0].plot(th, [r.y_perm[0] for r in res], color=COLORS[p], marker=MARKERS[p], ms=4, markevery=3, label=LABELS[p])
    ax[1].plot(th, [r.x_ret[0] for r in res], color=COLORS[p], marker=MARKERS[p], ms=4, markevery=3, label=LABELS[p])
for a, v, t in [(ax[0], 0.866, "H₂ in permeate"), (ax[1], 0.088, "H₂ in residue")]:
    a.plot([0.55], [v], marker="*", ms=14, color="#0b0b0b", ls="none", label="measured (Pan 1986)")
    a.set(xlabel="Stage cut θ", ylabel="Mole fraction", title=t)
ax[0].legend(fontsize=8);
"""),
    md(r"""
**Result.** The measured point lies closest to the **cross-flow** model, as Pan concluded, and far
from the co-current model the module was nominally run as. The toolkit's cross-flow model
over-predicts the separation slightly (88.6 % against 86.6 %). Pan reports the same deviation
above a stage cut of 0.55 and puts it down to back-diffusion in the porous support at high
recovery. The lesson for users: with high-flux asymmetric fibres, use `cross_flow` even when the
module is built co- or counter-current.
"""),
    md(r"""
## 2 · Merkel et al. (2010): flue-gas CO₂ capture, model results

Merkel et al. used the Polaris membrane (CO₂ 1000 GPU, CO₂/N₂ = 50) to show why flue-gas capture is
**pressure-ratio limited**. These are results from their own models, so the comparison checks that
the two sets of equations agree.

### 2a · The pressure-ratio limit (their Figs. 3 and 4)
The feed is 10 % CO₂ at 5 bar, the permeate is at 1 bar, and the stage cut is 1 %. The selectivity
is raised by lowering the N₂ permeance. However selective the membrane, the permeate cannot
exceed φ·x = 5 × 10 % = 50 % CO₂.
"""),
    code("""
alphas = np.logspace(0, 4, 25)
y, area = [], []
for a in alphas:
    r = cross_flow(Membrane({"CO2": 1000, "N2": 1000 / a}), [0.10, 0.90], 0.01, 5.0, 1.0, profile=False)
    y.append(r.y_perm[0]); area.append(r.area)
fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.5), constrained_layout=True)
ax[0].semilogx(alphas, y, color=COLORS["cross_flow"]); ax[0].axhline(0.5, color="#52514e", ls="--", lw=1)
ax[0].text(1.2, 0.47, "pressure-ratio limit φ·x = 0.50", color="#52514e", fontsize=8)
ax[0].set(xlabel="CO₂/N₂ selectivity", ylabel="Permeate CO₂", title="Merkel Fig. 4a: permeate purity")
ax[1].loglog(alphas, np.array(area) / area[0], color=COLORS["cross_flow"])
ax[1].set(xlabel="CO₂/N₂ selectivity", ylabel="Relative membrane area", title="Merkel Fig. 4b: area");
print(f"selectivity 1: {y[0]:.3f} (paper 0.10);  selectivity 10 000: {y[-1]:.3f} (paper: approaches 0.50)")
check("Merkel 2010 Fig. 4", "permeate CO2 at α = 1", 0.10, y[0], "model")
"""),
    md(r"""
Merkel et al. conclude that the optimum selectivity at φ = 5 lies between 20 and 40. Above that,
purity barely improves while the membrane area climbs steeply. The curves above show the same
behaviour.

### 2b · Cross-flow against counter-current (their Fig. 6)
The feed is 500 m³(STP)/s with 13 % CO₂, cleaned to 2.1 % CO₂ in the residue; the permeate is
under vacuum at 0.22 bar. The paper does not state the feed pressure. A blower of 7.2 MW at 80 %
efficiency raises 500 m³/s by about 10 %, so the feed is taken as **1.1 bar**, which gives the
pressure ratio of 5 used throughout the paper.
"""),
    code("""
from scipy.optimize import brentq
m = Membrane(MEMBRANE_PRESETS["Polaris CO2/N2 (Merkel 2010)"])
xf = FEED_PRESETS["flue gas, 13% CO2 (Merkel 2010)"]
F = 500 / 0.022414
ea = EnergyAssumptions(eta_vacuum=0.8, p_discharge=1.0, max_stage_ratio=10)   # single-stage vacuum pump
paper = {"cross_flow": (0.289, 11.0, 56.3), "countercurrent": (0.406, 6.8, 38.8)}
rows = {}
for p, (yp, ap, wp) in paper.items():
    th = brentq(lambda t: solve_stage(m, xf, t, 1.1, 0.22, p, F).x_ret[0] - 0.021, 0.05, 0.9, xtol=1e-7)
    r = solve_stage(m, xf, th, 1.1, 0.22, p, F)
    wv = stage_energy(r, ea)["vacuum_pump_kW"] / 1e3
    rows[LABELS[p]] = {"stage cut": th, "permeate CO2 (toolkit)": r.y_perm[0], "permeate CO2 (paper)": yp,
                       "area MM m2 (toolkit)": r.area / 1e6, "area MM m2 (paper)": ap,
                       "vacuum MW (toolkit)": wv, "vacuum MW (paper)": wp}
    check(f"Merkel 2010 Fig. 6, {LABELS[p]}", "permeate CO2", yp, r.y_perm[0], "model")
    check(f"Merkel 2010 Fig. 6, {LABELS[p]}", "area, million m2", ap, r.area / 1e6, "model")
    check(f"Merkel 2010 Fig. 6, {LABELS[p]}", "vacuum power, MW", wp, wv, "model")
pd.DataFrame(rows).T.round(3)
"""),
    md(r"""
**Result.** Permeate purity and membrane area agree within a few per cent, and counter-current
needs about 40 % less area than cross-flow in both models. The vacuum power comes out lower in the
toolkit. Merkel et al. do not give their compressor model, and their flue gas carries water and
oxygen that also have to be pumped, so this is the least certain comparison.

### 2c · Three membranes (their Table 1)
Feed 2.0 bar, permeate 0.2 bar, 90 % CO₂ capture, standard cross-flow modules. The paper's feed was
wet flue gas (11.6 % CO₂, 11 % H₂O, 4.2 % O₂); here it is treated as dry CO₂/N₂ with 13 % CO₂,
because permeances for water and oxygen are not given.
"""),
    code("""
paper = {"base case (1000 GPU, α 50)": (1000, 50, 0.46, 2.1),
         "membrane A (100 GPU, α 200)": (100, 200, 0.54, 59.0),
         "membrane B (1000 GPU, α 200)": (1000, 200, 0.55, 5.7)}
rows, base = {}, None
for name, (q, a, yp, ap) in paper.items():
    mm = Membrane({"CO2": q, "N2": q / a})
    th = brentq(lambda t: cross_flow(mm, xf, t, 2.0, 0.2, profile=False).recovery(0) - 0.90, 0.05, 0.95, xtol=1e-8)
    r = cross_flow(mm, xf, th, 2.0, 0.2, profile=False)
    base = base or r.area
    rows[name] = {"permeate CO2 (toolkit)": r.y_perm[0], "permeate CO2 (paper)": yp,
                  "area / base (toolkit)": r.area / base, "area / base (paper)": ap / 2.1}
    check("Merkel 2010 Table 1, " + name.split(" (")[0], "permeate CO2", yp, r.y_perm[0], "model")
pd.DataFrame(rows).T.round(3)
"""),
    md(r"""
**Result.** The ranking and the area ratios (about 28 and 2.7 times the base case) are reproduced.
Purities come out 2–3 points lower, as expected: in the paper's wet feed, water permeates fast and
acts as a built-in sweep, an effect Merkel et al. describe themselves. The central message holds:
the four-times more selective membrane B needs *more* area than the base case, because the
separation is pressure-ratio limited.

## 3 · Coker et al. (1998): a five-component refinery module, model results

The case is a shell-side-feed hollow-fibre module with permeate in the bores:
- 500 000 fibres, 300/150 µm outer/inner diameter, 0.8 m active length;
- 283.2 Nm³/h of hydrotreater purge gas at 42.4 bar and 50 °C, permeate at 7.9 bar;
- counter-current flow, with Hagen–Poiseuille pressure drop in the bores.

Coker et al. report a **total permeate recovery (stage cut) of 94.6 %**. They also note that at
this recovery H₂ is stripped from the feed early, so only about 20 % of the area does useful H₂
separation, and that methane passes through a concentration maximum.
"""),
    code("""
m = Membrane(MEMBRANE_PRESETS["Polysulfone-like H2/hydrocarbons (Coker 1998)"])
xf = FEED_PRESETS["refinery hydrotreater purge (Coker 1998)"]
mod = HollowFibreModule(500_000, length=0.8, d_inner=150e-6, d_outer=300e-6)
r = rate_module(m, xf, nm3h_to_mols(283.2), 42.4, 7.9, mod, "countercurrent", pressure_drop=True, T=323.15)
print(mod); print(f"stage cut: toolkit {r.theta:.4f}, paper 0.946;  bore pressure rise {r.extra['bore_dp_bar']:.3f} bar")
check("Coker 1998 refinery module", "stage cut", 0.946, r.theta, "model")
pr = r.profile
fig, ax = plt.subplots(figsize=(6.2, 4), constrained_layout=True)
for i, c in enumerate(m.components):
    ax.plot(pr["area_fraction"], pr["x"][i], color=["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"][i], label=c)
ax.axvline(0.2, color="#52514e", ls=":", lw=1); ax.text(0.21, 0.62, "≈ 20 % of the area", color="#52514e", fontsize=8)
ax.set(xlabel="Fraction of fibre length from the feed end", ylabel="Feed-side mole fraction",
       title="Coker et al. (1998) refinery case", xlim=(0, 1), ylim=(0, None)); ax.legend();
h2 = pr["x"][0]
print(f"feed-side H2 falls below 1 % at {pr['area_fraction'][np.argmax(h2 < 0.01)]:.2f} of the length")
"""),
    md(r"""
**Result.** The stage cut matches (94.65 % against 94.6 %), H₂ is gone from the feed side within
the first ~20 % of the fibre, and methane peaks partway along, as Coker et al. describe. Two notes
on the scanned paper: the module area must be 377 m² (π × 300 µm × 0.8 m × 500 000; the "317" in
the scan is a misread), and the component order of Table 3 was confirmed by this run.

## 4 · Makaruk et al. (2010): two-stage biogas cascade, model results

Makaruk et al. recommend a **two-stage cascade with recycle**: the retentate of stage 1 is polished
in stage 2, whose permeate returns to the compressor. In the toolkit this is
`two_stage_stripping`. Their basis:
- 1000 Nm³/h of biogas with 60 % CH₄, a product with 98 % CH₄;
- 10 bar feed, counter-current modules;
- CO₂ permeance 100 GPU;
- a two-stage compressor with 80 % isentropic and 90 % mechanical efficiency.

For the two heat-demand cases in the paper, 5 % and 15 %, the methane recoveries are 93.8 % and
81.2 %. Their headline figure is a specific power of **about 0.3 kWh per Nm³ of product**.
"""),
    code("""
from scipy.optimize import least_squares
xf = FEED_PRESETS["biogas, 60% CH4 (Makaruk 2010)"]; F = nm3h_to_mols(1000)
ea = EnergyAssumptions(eta_compressor=0.8 * 0.9)

def cascade(alpha, th, pattern):
    mm = Membrane({"CO2": 100.0, "CH4": 100.0 / alpha})
    return two_stage_stripping(mm, xf, F, StageSpec(th[0], 10, 1, pattern), StageSpec(th[1], 10, 1, pattern), ea=ea)

def design(alpha, recovery, pattern, x0=(0.3, 0.3)):
    # stage cuts that give 98 % CH4 at the requested recovery
    f = lambda th: (lambda r: [r.purity("retentate 2", "CH4") - 0.98,
                               r.recovery("retentate 2", "CH4") - recovery])(cascade(alpha, th, pattern))
    th = least_squares(f, x0, bounds=([0.05, 0.005], [0.8, 0.9]), xtol=1e-7, ftol=1e-7, diff_step=1e-3).x
    r = cascade(alpha, th, pattern)
    return dict(pattern=pattern, alpha=alpha, recovery=recovery, theta1=th[0], theta2=th[1],
                purity=r.purity("retentate 2", "CH4"), area_m2=r.total_area,
                kWh_per_Nm3=r.total_power_kW / mols_to_nm3h(r.streams["retentate 2"].sum()),
                recycle_ratio=r.recycle_ratio())

# selectivity sweep with cross-flow stages (fast) ...
mk = pd.DataFrame([design(a, rec, "cross_flow") for rec in (0.938, 0.812) for a in (12, 22, 35, 50)])
# ... and the alpha = 35 cases with counter-current stages, as in the paper (a few minutes)
cc = pd.DataFrame([design(35, rec, "countercurrent",
                          x0=mk[(mk.alpha == 35) & (mk.recovery == rec)][["theta1", "theta2"]].iloc[0].values)
                   for rec in (0.938, 0.812)])
for _, row in cc.iterrows():
    check(f"Makaruk 2010, α = 35, recovery {row.recovery:.1%}", "kWh per Nm3 product", 0.3,
          row.kWh_per_Nm3, "model (approximate)")
pd.concat([mk, cc]).round(3)
"""),
    code("""
fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.5), constrained_layout=True)
for rec, ls in [(0.938, "-"), (0.812, "--")]:
    d = mk[mk.recovery == rec]
    ax[0].plot(d.alpha, d.kWh_per_Nm3, color=COLORS["cross_flow"], ls=ls, marker="s", ms=4, label=f"cross-flow, recovery {rec:.1%}")
    ax[1].plot(d.alpha, d.area_m2, color=COLORS["cross_flow"], ls=ls, marker="s", ms=4, label=f"cross-flow, recovery {rec:.1%}")
ax[0].plot(cc.alpha, cc.kWh_per_Nm3, color=COLORS["countercurrent"], ls="none", marker="D", ms=6, label="counter-current, α = 35")
ax[1].plot(cc.alpha, cc.area_m2, color=COLORS["countercurrent"], ls="none", marker="D", ms=6, label="counter-current, α = 35")
ax[0].axhline(0.3, color="#52514e", lw=1, ls=":"); ax[0].text(13, 0.305, "≈ 0.3 kWh/Nm³ (Makaruk 2010)", fontsize=8, color="#52514e")
ax[0].set(xlabel="CO₂/CH₄ selectivity", ylabel="kWh per Nm³ product", title="Specific compression power (cf. their Fig. 15)")
ax[1].set(xlabel="CO₂/CH₄ selectivity", ylabel="Total membrane area (m²)", title="Membrane area (cf. their Fig. 16)")
ax[0].legend(fontsize=8); ax[1].legend(fontsize=8);
"""),
    md(r"""
**Result.** Both of Makaruk's design findings are reproduced:
- **Optimum selectivity:** the selectivity that needs the least membrane area comes out at
  **α ≈ 22 for 93.8 % recovery and α ≈ 12 for 81.2 %**, the values reported in their Fig. 16.
  Area rises again at higher selectivity because, with the CO₂ permeance fixed, a more selective
  membrane lets less gas through per square metre.
- **Specific compression power:** 0.23–0.30 kWh per Nm³ of product (0.24 with counter-current
  modules at α = 35), close to their "around 0.3 kWh/m³". Like their Fig. 15, it falls with
  selectivity and flattens above about 30.
- **Near-single-stage limit:** at 81.2 % recovery and α = 35 the optimiser drives the second
  stage almost to zero. Makaruk et al. note the same: for the higher heat demand a single stage
  is sufficient.

Their figures can only be read as trends from the scan, so these checks are approximate.

## 5 · Feng et al. (1999): air separation, measured

Feng et al. tested a module of 368 asymmetric fibres, 25 cm long and 160/80 µm in diameter, with
bore-side feed in counter-current flow. Their feed was 20.5 % O₂ air at 100 psig (690 kPa gauge),
with the permeate at local atmospheric pressure, 70 cmHg (93.3 kPa). The pure-gas permeances
follow Arrhenius laws with activation energies of 19.3 kJ/mol for O₂ and 27.6 kJ/mol for N₂.

The measured points are only in figures, but the text states three quantitative findings, which
are checked here:
1. Nitrogen productivity roughly **halves** from 95 % to 98 % N₂ purity.
2. From 23 to 45 °C at 95 % N₂, productivity rises by about **75 %** and recovery falls by about
   **8 points**. At 35 % O₂, productivity about **doubles** and recovery falls by about **6 points**.
3. At 95 % N₂, adding **50 psi** to the feed is about **equivalent** to pulling a **57 cmHg vacuum**
   on the permeate.

Productivity is the product flow per unit membrane area.
"""),
    code("""
from gasmem.properties import GPU, R
Q0 = {"O2": 30.78e-10 / GPU, "N2": 5.7e-10 / GPU}; E = {"O2": 19.3e3, "N2": 27.6e3}
def feng_membrane(T_C):
    T = T_C + 273.15
    return Membrane({g: Q0[g] * np.exp(-E[g] / R * (1 / T - 1 / 296.15)) for g in Q0})
xf = [0.205, 0.795]; patm = 0.933; p100 = 6.895 + patm

def n2_case(purity, T_C=23, ph=p100, pl=patm):
    mm = feng_membrane(T_C)
    th = brentq(lambda t: countercurrent(mm, xf, t, ph, pl, profile=False).x_ret[1] - purity, 0.25, 0.9, xtol=1e-8)
    r = countercurrent(mm, xf, th, ph, pl, profile=False)
    return r.retentate_flow * r.x_ret[1] / r.area, r.recovery("N2", "retentate")

def o2_case(purity, T_C=23):
    mm = feng_membrane(T_C)
    th = brentq(lambda t: countercurrent(mm, xf, t, p100, patm, profile=False).y_perm[0] - purity, 0.02, 0.6, xtol=1e-8)
    r = countercurrent(mm, xf, th, p100, patm, profile=False)
    return r.permeate_flow * r.y_perm[0] / r.area, r.recovery("O2")

p95, r95 = n2_case(0.95); p98, _ = n2_case(0.98); p95h, r95h = n2_case(0.95, 45)
o35, ro35 = o2_case(0.35); o35h, ro35h = o2_case(0.35, 45)
pa, _ = n2_case(0.95, ph=p100 + 3.447); pb, _ = n2_case(0.95, pl=patm - 0.760)
feng = pd.DataFrame([
    ("productivity ratio, 98 % / 95 % N2", 0.5, p98 / p95),
    ("23→45 °C, 95 % N2: productivity gain", 0.75, p95h / p95 - 1),
    ("23→45 °C, 95 % N2: recovery change (points)", -8, (r95h - r95) * 100),
    ("23→45 °C, 35 % O2: productivity ratio", 2.0, o35h / o35),
    ("23→45 °C, 35 % O2: recovery change (points)", -6, (ro35h - ro35) * 100),
    ("+50 psi feed vs 57 cmHg vacuum: productivity ratio", 1.0, pa / pb),
], columns=["finding", "Feng et al. (approx.)", "toolkit"]).set_index("finding")
for f, row in feng.iterrows():
    check("Feng 1999", f, row.iloc[0], row.iloc[1], "measured (approximate)")
feng.round(2)
"""),
    md(r"""
**Result.** All three findings are reproduced in direction and roughly in size by an ideal
counter-current model with constant pure-gas permeances. The productivity gain on heating comes out
smaller than measured (+57 % against +75 %). Feng et al. note that their model also slightly
over-predicts purity at high stage cut, which they attribute to concentration polarisation in the
porous support. The toolkit does not model that effect, nor the feed-side pressure drop of
bore-side feed.

## Summary
"""),
    code("""
summary = pd.DataFrame(checks)
def diff(row):
    try:
        return f"{(row.toolkit - row.published) / abs(row.published) * 100:+.0f} %"
    except ZeroDivisionError:
        return ""
summary["difference"] = summary.apply(diff, axis=1)
summary.style.format({"published": "{:.3g}", "toolkit": "{:.3g}"}).hide(axis="index")
"""),
    md(r"""
**Overall.** Where the published numbers come from models with the same assumptions as the
toolkit, agreement is within a few per cent: Merkel's purities and areas, and Coker's stage cut.
Against measurements, the ideal models capture the trends and land within a few percentage points
of composition, provided the right flow pattern is chosen (cross-flow for Pan's high-flux
asymmetric fibres). The remaining gaps all trace back to effects the toolkit deliberately leaves
out:
- concentration polarisation and back-diffusion in the porous support;
- mixed-gas permeances that differ from pure-gas values;
- water and oxygen in real flue gas;
- feed-side pressure drop.
"""),
]

# ---------------------------------------------------------------------------
NOTEBOOKS["08_experimental_data_point_by_point.ipynb"] = [
    md(r"""
# 8 · Point by point against digitised experimental data

Notebook 07 compared single numbers quoted in the papers' text. This notebook goes further: the
**measured points in the papers' figures** were digitised, and every point is compared with the
toolkit at the paper's conditions. That gives 274 measured points from two papers and 64
model-curve points from a third:

| Paper | Figures | Points | What they are |
|---|---|---|---|
| Pan (1986) | 2, 3 | 60 | measured; H₂ recovery with **pure-gas** permeances (a true prediction) |
| Pan (1986) | 6, 7 | 55 | measured; acid gases, with permeances Pan **fitted to one run** |
| Feng et al. (1999) | 4, 5 | 78 | measured; air separation, four flow arrangements |
| Feng et al. (1999) | 13, 14 | 81 | measured; recovery and **absolute productivity** at 23 and 45 °C |
| Makaruk et al. (2010) | 15, 16 | 64 | their **model** curves; specific power and membrane area |

**How the points were read.** The figures were rendered at 300–400 dpi. Axes were calibrated
from the detected tick marks, and marker centres were found by image processing; markers touching
a curve were located by hand on magnified crops. Every point is stored with its pixel coordinates
in `gasmem/data/`, and `tools/digitised_data.py` documents each calibration. Reading uncertainty is
about ±3 px: roughly ±0.1 % in composition and ±0.002 in stage cut.
"""),
    code(SETUP),
    code("""
from gasmem.datasets import load, available, DATA as DATA_DIR
from gasmem.properties import GPU, R, NM3_PER_MOL
from scipy.optimize import brentq

def stats(df, by):
    g = df.groupby(by)
    return pd.DataFrame({"points": g.size(), "mean error": g.err.mean(),
                         "RMS error": g.err.apply(lambda e: float(np.sqrt((e ** 2).mean())))})

def predict(df, solve, value):
    \"\"\"Add a 'pred' column: solve(x) -> result (cached per x), value(result, row) -> number.\"\"\"
    cache = {}
    out = []
    for row in df.itertuples():
        k = round(row.x, 5)
        if k not in cache:
            cache[k] = solve(row.x)
        out.append(value(cache[k], row))
    return df.assign(pred=out, err=lambda t: t.pred - t.y)

summary = []
available()
"""),
    md(r"""
## 1 · Pan (1986), Figs. 2–3: hydrogen recovery, a true prediction

Cellulose acetate hollow fibres, ammonia purge gas, 6964 kPa / 1123 kPa, 25 °C. The permeances
are Pan's **pure-gas** values, so nothing is fitted to these runs. Filled markers are runs with
counter-current flow, open markers co-current runs.
"""),
    code("""
pan = load("Pan1986")
m = Membrane(MEMBRANE_PRESETS["Cellulose acetate H2 recovery (Pan 1986)"])
xf = FEED_PRESETS["ammonia purge gas, NH3-free (Pan 1986)"]
d23 = pan[pan.figure.isin(["Fig. 2", "Fig. 3"])]

def value_pan(r, row):
    if row.quantity == "H2 recovery %":
        return 100 * r.recovery("H2")
    gas = row.quantity.split()[1]
    return 100 * r.y_perm[r.components.index(gas)]

res = {p: predict(d23, lambda t, p=p: solve_stage(m, xf, t, 69.64, 11.23, p), value_pan)
       for p in ["cross_flow", "countercurrent", "cocurrent"]}
tab = pd.concat({LABELS[p]: stats(r, ["quantity", "series"])["RMS error"] for p, r in res.items()}, axis=1)
tab.round(2)
"""),
    code("""
th = np.linspace(0.30, 0.62, 25)
curves = {p: [solve_stage(m, xf, t, 69.64, 11.23, p) for t in th] for p in ["cross_flow", "countercurrent", "cocurrent"]}
panels = [("permeate H2 %", lambda r: 100 * r.y_perm[0]), ("H2 recovery %", lambda r: 100 * r.recovery("H2")),
          ("permeate N2 %", lambda r: 100 * r.y_perm[1]), ("permeate CH4 %", lambda r: 100 * r.y_perm[2]),
          ("permeate Ar %", lambda r: 100 * r.y_perm[3])]
fig, axes = plt.subplots(1, 5, figsize=(17, 3.4), constrained_layout=True)
for ax, (q, f) in zip(axes, panels):
    for p, rs in curves.items():
        ax.plot(th, [f(r) for r in rs], color=COLORS[p], label=LABELS[p])
    for series, mk in [("countercurrent", dict(marker="o", color="#0b0b0b")),
                       ("cocurrent", dict(marker="o", mfc="white", mec="#0b0b0b"))]:
        s = d23[(d23.quantity == q) & (d23.series == series)]
        ax.plot(s.x, s.y, ls="none", ms=6, label=f"measured, {series} run", **mk)
    ax.set(title=q, xlabel="Stage cut θ")
axes[0].legend(fontsize=7);
"""),
    code("""
# Is the measured recovery consistent with the measured purity? Mass balance: R = θ·y/x_F
f2 = d23[d23.figure == "Fig. 2"]
both = f2.pivot_table(index=[f2.series, f2.x.round(2)], columns="quantity", values="y")
both["recovery from mass balance"] = 100 * both.index.get_level_values(1) * both["permeate H2 %"] / 51.78
both["difference"] = both["H2 recovery %"] - both["recovery from mass balance"]
print(f"plotted recovery exceeds the mass-balance value by {both['difference'].mean():.1f} points on average")
cf = stats(res["cross_flow"], ["quantity", "series"])
for (q, sr), row in cf.iterrows():
    summary.append(dict(paper="Pan 1986 (prediction)", figure="2–3", quantity=q, series=sr,
                        points=row.points, RMS=row["RMS error"], model="cross-flow"))
"""),
    md(r"""
**Result.**
- **Compositions:** the cross-flow model, which Pan argues is the right one for high-flux
  asymmetric fibres, predicts the permeate within about 1 % H₂ and within a few tenths of a
  per cent for N₂, CH₄ and Ar. No parameter was fitted.
- **Co-current runs:** these are clearly *not* described by the homogeneous co-current model.
  This is Pan's central experimental finding, reproduced.
- **H₂ recovery:** under-predicted by about 3 points with every model. The last cell shows why:
  the plotted recoveries exceed what the plotted purities imply through the mass balance by
  about 3 points. The offset is in the data, not the model.

## 2 · Pan (1986), Figs. 6–7: acid gases, with fitted permeances

Pan found that mixed-gas permeances differ from pure-gas values (CH₄ and N₂ permeate about
twice as fast, because CO₂ and H₂S plasticise the polymer). For these runs he therefore fitted
the permeances to **one** run at an arbitrary stage cut and predicted the rest. The same
permeances are used here, so this tests the shape of the stage-cut dependence.
"""),
    code("""
m6 = Membrane(MEMBRANE_PRESETS["Cellulose acetate CO2/hydrocarbons (Pan 1986)"])
x6 = FEED_PRESETS["acid gas with hydrocarbons (Pan 1986)"]
m7 = Membrane(MEMBRANE_PRESETS["Cellulose acetate H2S/CO2 sour gas (Pan 1986)"])
x7 = FEED_PRESETS["sour gas, 22% H2S (Pan 1986)"]
cases = {"Fig. 6": (m6, x6, 35.28, 0.928), "Fig. 7": (m7, x7, 52.45, 0.928)}
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), constrained_layout=True)
pal = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
rows = []
for ax, (fg, (mm, xx, ph, pl)) in zip(axes, cases.items()):
    d = pan[pan.figure == fg]
    r = predict(d, lambda t: solve_stage(mm, xx, t, ph, pl, "cross_flow"),
                lambda r, row: 100 * r.y_perm[r.components.index(row.quantity.split()[1])])
    rows.append(stats(r, ["figure", "quantity"]))
    tt = np.linspace(d.x.min() - 0.03, d.x.max() + 0.03, 25)
    rs = [solve_stage(mm, xx, t, ph, pl, "cross_flow") for t in tt]
    for i, q in enumerate(sorted(d.quantity.unique())):
        gas = q.split()[1]; k = mm.components.index(gas)
        ax.plot(tt, [100 * r.y_perm[k] for r in rs], color=pal[i], label=gas)
        s = d[d.quantity == q]; ax.plot(s.x, s.y, ls="none", marker="o", ms=5, color=pal[i], mec="#0b0b0b")
    ax.set(title=f"Pan (1986) {fg}", xlabel="Stage cut θ", ylabel="Permeate mole %", yscale="log")
    ax.legend(fontsize=8)
t67 = pd.concat(rows)
for (fg, q), row in t67.iterrows():
    summary.append(dict(paper="Pan 1986 (fitted)", figure=fg.split()[1], quantity=q, series="countercurrent",
                        points=row.points, RMS=row["RMS error"], model="cross-flow"))
t67.round(2)
"""),
    md(r"""
**Result.** All components agree within 0.1–1 % RMS. Once the mixed-gas permeances are known,
the stage-cut dependence is captured very well. C₃H₈ (log scale, left) looks worse than it is: its
points lie only 7–30 px above zero on a 0–12 % axis, so a ±3 px reading uncertainty is ±0.05 %,
comparable to the values themselves.

## 3 · Feng et al. (1999), Figs. 4–5: four flow arrangements

Asymmetric hollow fibres, air at 690 kPa(g), permeate at local atmospheric pressure (70 cmHg),
23 °C, with the pure-gas permeances. Feng et al. ran the same module in four ways: feed in the
bores or on the shell side, each counter-current and co-current.
"""),
    code("""
feng = load("Feng1999")
mF = Membrane(MEMBRANE_PRESETS["Asymmetric hollow fibre air (Feng 1999)"])
xa = [0.205, 0.795]; ph, pl = 6.895 + 0.933, 0.933
d45 = feng[feng.figure.isin(["Fig. 4", "Fig. 5"])]
val = lambda r, row: 100 * (r.x_ret[1] if "N2" in row.quantity else r.y_perm[0])
resF = {p: predict(d45, lambda t, p=p: solve_stage(mF, xa, t, ph, pl, p), val) for p in ["countercurrent", "cocurrent"]}
tabF = pd.concat({LABELS[p]: stats(r, ["quantity", "series"])["RMS error"] for p, r in resF.items()}, axis=1)
tabF.round(2)
"""),
    code("""
tt = np.linspace(0.25, 0.86, 25)
cv = {p: [solve_stage(mF, xa, t, ph, pl, p) for t in tt] for p in ["countercurrent", "cocurrent"]}
style = {"bore-side feed, countercurrent": dict(marker="o", mfc="white", mec="#0b0b0b"),
         "shell-side feed, countercurrent": dict(marker="o", color="#0b0b0b", ms=4),
         "bore-side feed, cocurrent": dict(marker="s", mfc="white", mec="#0b0b0b"),
         "shell-side feed, cocurrent": dict(marker="s", color="#0b0b0b", ms=4)}
fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
for ax, q, f in [(axes[0], "residue N2 %", lambda r: 100 * r.x_ret[1]), (axes[1], "permeate O2 %", lambda r: 100 * r.y_perm[0])]:
    for p, rs in cv.items():
        ax.plot(tt, [f(r) for r in rs], color=COLORS[p], label=f"toolkit, {LABELS[p].lower()}")
    for sr, stl in style.items():
        s = d45[(d45.quantity == q) & (d45.series == sr)]
        ax.plot(s.x, s.y, ls="none", label=sr, **({"ms": 6} | stl))
    ax.set(title=f"Feng et al. (1999): {q}", xlabel="Stage cut θ")
axes[0].legend(fontsize=7);
for sr in style:
    p = "countercurrent" if "countercurrent" in sr else "cocurrent"
    s = resF[p][resF[p].series == sr]
    for q, g in s.groupby("quantity"):
        summary.append(dict(paper="Feng 1999", figure="4–5", quantity=q, series=sr, points=len(g),
                            RMS=float(np.sqrt((g.err ** 2).mean())), model=LABELS[p].lower()))
"""),
    md(r"""
**Result.**
- **Matching flow model:** with the flow model that matches the module, the bore-side feed runs
  agree within **0.1–0.3 %** (co-current) and **0.3–0.4 %** (counter-current), using pure-gas
  permeances and no fitting. Unlike Pan's high-flux fibres, these fibres clearly follow
  counter-current and co-current behaviour, as Feng et al. concluded.
- **Shell-side feed:** these runs perform about 0.4 points worse than predicted. They carry the
  permeate inside 80 µm bores. Running `rate_module` with the actual geometry (368 fibres, 25 cm,
  80/160 µm) gives a bore pressure rise of only about 0.04 bar, which closes very little of this
  gap. The likelier cause is the uneven shell-side flow distribution that Feng et al. mention,
  and the toolkit does not model it.

## 4 · Feng et al. (1999), Figs. 13–14: recovery and absolute productivity

Productivity is product flow per square metre of membrane, so it tests the permeances' absolute
values, not just their ratio. The 45 °C permeances come from Feng's activation energies
(19.3 kJ/mol for O₂, 27.6 kJ/mol for N₂).
"""),
    code("""
Q0 = {"O2": 30.78e-10 / GPU, "N2": 5.7e-10 / GPU}; E = {"O2": 19.3e3, "N2": 27.6e3}
memT = lambda TC: Membrane({g: Q0[g] * np.exp(-E[g] / R * (1 / (TC + 273.15) - 1 / 296.15)) for g in Q0})
to_nm3h = NM3_PER_MOL * 3600

def feng_point(row):
    mm = memT(int(row.series.split()[0]))
    if "N2" in row.quantity:
        t = brentq(lambda t: countercurrent(mm, xa, t, ph, pl, profile=False).x_ret[1] - row.x / 100, 0.2, 0.97, xtol=1e-9)
        r = countercurrent(mm, xa, t, ph, pl, profile=False)
        return 100 * r.recovery("N2", "retentate") if "recovery" in row.quantity else r.retentate_flow * r.x_ret[1] / r.area * to_nm3h
    t = brentq(lambda t: countercurrent(mm, xa, t, ph, pl, profile=False).y_perm[0] - row.x / 100, 0.01, 0.97, xtol=1e-9)
    r = countercurrent(mm, xa, t, ph, pl, profile=False)
    return 100 * r.recovery("O2") if "recovery" in row.quantity else r.permeate_flow * r.y_perm[0] / r.area * to_nm3h

d1314 = feng[feng.figure.isin(["Fig. 13", "Fig. 14"])].copy()
d1314["pred"] = [feng_point(r) for r in d1314.itertuples()]
d1314["err"] = d1314.pred - d1314.y
d1314["range"] = np.where((d1314.figure == "Fig. 13") & (d1314.x > 97.5), "N2 above 97.5 %", "rest")
t1314 = stats(d1314, ["quantity", "series", "range"])
t1314.round(3)
"""),
    code("""
fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)
for col, (fg, gas, xlab) in enumerate([("Fig. 13", "N2", "Residue N₂ (mol %)"), ("Fig. 14", "O2", "Permeate O₂ (mol %)")]):
    d = d1314[d1314.figure == fg]
    for rowi, kind in enumerate(["recovery", "productivity"]):
        ax = axes[rowi, col]
        for TC, c in [(23, COLORS["countercurrent"]), (45, COLORS["cross_flow"])]:
            s = d[(d.series == f"{TC} C") & d.quantity.str.contains(kind)].sort_values("x")
            ax.plot(s.x, s.y, ls="none", marker="o", ms=6, color=c, mec="#0b0b0b", label=f"measured {TC} °C")
            ax.plot(s.x, s.pred, color=c, label=f"toolkit {TC} °C")
        ax.set(xlabel=xlab, title=f"{gas} {kind}", ylabel=("%" if kind == "recovery" else "Nm³/(m² h)"))
        ax.legend(fontsize=7)
for (q, sr, rg), row in t1314.iterrows():
    summary.append(dict(paper="Feng 1999", figure="13–14", quantity=q + ("" if rg == "rest" else f" ({rg})"),
                        series=sr, points=row.points, RMS=row["RMS error"], model="counter-current"))
"""),
    md(r"""
**Result.**
- **O₂ enrichment:** recovery and absolute productivity agree well at both temperatures. With
  pure-gas permeances and Arrhenius temperature dependence, productivity is within a few per cent.
- **N₂ up to about 97.5 % purity:** recovery is within 2–3 points, productivity within about
  0.006 Nm³/(m² h).
- **N₂ above 98 %:** the toolkit over-predicts both recovery and productivity. Feng et al.'s own
  model does the same, visible as their curve passing above the data in Fig. 13. They attribute it
  to concentration polarisation in the porous support at high stage cut, which ideal models
  (theirs and the toolkit's) leave out. This is a real limitation to state when the toolkit is used
  for high-purity nitrogen.

## 5 · Makaruk et al. (2010), Figs. 15–16: comparison with their model

These are model curves, not measurements, so the comparison checks that two independent
implementations of the same counter-current cascade agree. The toolkit runs take about half an
hour, so they are stored in `gasmem/data/makaruk2010_toolkit_runs.csv`; regenerate them with
`python tools/run_makaruk_cascades.py`.
"""),
    code("""
mak = load("Makaruk2010")
runs = pd.read_csv(DATA_DIR / "makaruk2010_toolkit_runs.csv")
key = {"recovery 93.8 % (5 % heat)": 0.938, "recovery 81.2 % (15 % heat)": 0.812}
fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
cmp_rows = []
for ax, fg, col, ylab in [(axes[0], "Fig. 15", "kWh_per_Nm3", "kWh per Nm³ product"), (axes[1], "Fig. 16", "area_m2", "Total membrane area (m²)")]:
    for (sr, rec), ls in zip(key.items(), ["-", "--"]):
        s = mak[(mak.figure == fg) & (mak.series == sr)].sort_values("x")
        xs, ys = s.x.to_numpy(), s.y.to_numpy()
        gap = np.where((np.diff(xs) > 5) & (xs[:-1] < 20))[0]   # steep part hidden by a label
        xs, ys = np.insert(xs, gap + 1, np.nan), np.insert(ys, gap + 1, np.nan)
        ax.plot(xs, ys, color="#52514e", ls=ls, label=f"Makaruk et al., {sr.split(' (')[0]}")
        for p, mkr in [("countercurrent", "D"), ("cross_flow", "s")]:
            rr = runs[(runs.recovery_target == rec) & (runs.pattern == p)].sort_values("alpha")
            ax.plot(rr.alpha, rr[col], ls="none", marker=mkr, ms=6, color=COLORS[p],
                    label=f"toolkit {LABELS[p].lower()}" if rec == 0.938 else None)
            if p == "countercurrent":
                for _, q in rr.iterrows():
                    if np.min(np.abs(s.x - q.alpha)) > 1.5:     # curve hidden by a label there
                        continue
                    ref = np.interp(q.alpha, s.x, s.y)
                    cmp_rows.append(dict(figure=fg, recovery=rec, alpha=q.alpha, Makaruk=ref, toolkit=q[col],
                                         difference_pct=100 * (q[col] - ref) / ref,
                                         recovery_reached=q.recovery, purity_reached=q.purity))
    ax.set(xlabel="CO₂/CH₄ selectivity", ylabel=ylab, title=f"Makaruk et al. (2010) {fg}")
    ax.legend(fontsize=7)
cmpm = pd.DataFrame(cmp_rows)
cmpm.round(3)
"""),
    code("""
ok = cmpm[((cmpm.recovery_reached - cmpm.recovery).abs() < 0.003) & (cmpm.purity_reached < 0.985)]
for fg, g in ok.groupby("figure"):
    summary.append(dict(paper="Makaruk 2010 (model)", figure=fg.split()[1],
                        quantity="power" if fg == "Fig. 15" else "area", series="both recoveries",
                        points=len(g), RMS=float(np.sqrt((g.difference_pct ** 2).mean())),
                        model="counter-current, % difference"))
print(f"{len(cmpm) - len(ok)} runs could not reach the target recovery exactly with two stages "
      "(very high selectivity: a single stage already exceeds it; Makaruk et al. note that feed bypass is then needed)")
"""),
    md(r"""
**Result.** Membrane areas agree within a few per cent across the whole selectivity range, and the
area minima fall at the same selectivities (about 22 and about 12). The specific power agrees in
shape; the toolkit's values are slightly higher, consistent with a small difference in compressor
modelling. Makaruk et al. give only efficiencies and the interstage pressure.

## Summary
"""),
    code("""
S = pd.DataFrame(summary)
S["RMS"] = S["RMS"].round(3)
S.style.hide(axis="index").format({"RMS": "{:.3g}"})
"""),
    md(r"""
**What this shows.** With the published permeances and no fitting:
- the toolkit reproduces Pan's and Feng's measured compositions to within about **0.1–1.5 %**
  across the full stage-cut range, provided the flow model matches the module (cross-flow for
  Pan's high-flux asymmetric fibres, counter-current or co-current for Feng's);
- it reproduces Feng's **absolute** productivities;
- it agrees with Makaruk's cascade model within a few per cent.

The deviations that remain come from physics the toolkit deliberately omits: concentration
polarisation at high stage cut (Feng, N₂ above 98 %), shell-side flow maldistribution (Feng), and
mixed-gas permeances that differ from pure-gas values (Pan's acid gases).
"""),
]


# ---------------------------------------------------------------------------
NOTEBOOKS["09_design_optimisation.ipynb"] = [
    md(r"""
# 09 · Design optimisation: choosing stage cuts and pressures

Earlier notebooks set the stage cuts by hand. Here the toolkit chooses them.

* `stage_cut_for_spec` finds the stage cut of **one stage** that meets one specification
  (a purity or a recovery). It is a scalar root-find and takes about a second.
* `optimise` handles **any layout and several variables**. It minimises cost per tonne,
  membrane area, power or specific energy, subject to a minimum purity and recovery of a product.

The case is biogas upgrading at Makaruk et al. (2010) conditions: 40 % CO₂ / 60 % CH₄,
1 000 Nm³/h, polyimide membrane with α = 35, target 98 % CH₄ in the retentate product.
"""),
    code(SETUP),
    code("""
m  = Membrane(MEMBRANE_PRESETS["Polyimide CO2/CH4, a=35 (Makaruk 2010)"])
xf = FEED_PRESETS["biogas, 60% CH4 (Makaruk 2010)"]
F  = nm3h_to_mols(1000)
"""),
    md(r"""
## 1. One stage, one specification

What stage cut gives 98 % CH₄ in the retentate, for each flow pattern? The answer fixes the
methane recovery, which a single stage cannot choose independently.
"""),
    code("""
rows = []
for pat in PATTERNS:
    try:
        r = stage_cut_for_spec(m, xf, "CH4", 0.98, p_high=10, p_low=1, pattern=pat,
                               stream="retentate", spec="purity", feed_flow=F)
        rows.append({"pattern": LABELS[pat], "stage cut": r.theta,
                     "CH4 recovery": r.recovery("CH4", "retentate"), "area m2": r.area})
    except InfeasibleError as e:
        rows.append({"pattern": LABELS[pat], "note": str(e)})
pd.DataFrame(rows).set_index("pattern")
"""),
    md(r"""
Only cross-flow and counter-current stages reach 98 % at a pressure ratio of 10. In a well-mixed
stage the whole permeate side sees the retentate composition, and in a co-current stage the
permeate-side CO₂ builds up along the module; in both, the driving force for CO₂ vanishes before the
retentate is pure enough. `stage_cut_for_spec` reports the reachable range instead of failing
silently.

Even the stages that reach the target lose 18–24 % of the methane. A recovery target therefore
needs a cascade, and a cascade has several stage cuts and pressures to choose: that is the job of
`optimise`.

## 2. Several variables: a two-stage stripping cascade

Write a small function that builds the flowsheet from the variables you want to choose. Here:
the two stage cuts and the feed pressure, with cross-flow stages and a 1 bar permeate.
"""),
    code("""
def design(theta1, theta2, p_high):
    return two_stage_stripping(m, xf, F,
                               StageSpec(theta1, p_high, 1.0, "cross_flow"),
                               StageSpec(theta2, p_high, 1.0, "cross_flow"))

bounds = {"theta1": (0.15, 0.6), "theta2": (0.15, 0.7), "p_high": (5.0, 20.0)}
best = optimise(design, bounds, product="slow-gas product", component="CH4",
                purity=0.98, recovery=0.97, objective="cost")
print(best.summary())
"""),
    md(r"""
`optimise` first scans the design space with a quasi-random (Sobol) sample, then refines the best
point that meets the targets with a constrained local optimiser. Every evaluation is kept in
`best.history`, so the scan doubles as a map of the trade-off:
"""),
    code("""
h = best.history[best.history["solved"]]
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
for a, col, lab in ((ax[0], "purity", "CH₄ purity"), (ax[1], "recovery", "CH₄ recovery")):
    sc = a.scatter(h["theta1"], h["theta2"], c=h[col], cmap="viridis", s=28)
    a.plot(best.variables["theta1"], best.variables["theta2"], "*", color="#e34948", ms=16,
           mec="white")
    a.set_xlabel("stage cut 1"); a.set_ylabel("stage cut 2"); a.set_title(f"{lab} (★ optimum)")
    fig.colorbar(sc, ax=a)
plt.tight_layout()
"""),
    md(r"""
## 3. The optimum depends on the prices

With two active constraints (purity and recovery) and three variables, one degree of freedom is
left, and the cost assumptions decide where it goes. Cheap electricity favours a high feed
pressure (less membrane); expensive electricity favours a lower pressure and more membrane.

The defaults in `CostAssumptions` are placeholders, so treat the numbers as an illustration of the
trend, not as prices. Each optimisation below starts from the previous optimum (`x0`) with a short
scan, which keeps it quick.
"""),
    code("""
rows, x0 = [], best.variables
for price in (0.05, 0.10, 0.20, 0.30):
    ca = CostAssumptions(electricity_per_kWh=price)
    o = optimise(design, bounds, "slow-gas product", "CH4", purity=0.98, recovery=0.97,
                 objective="cost", ca=ca, x0=x0, n_scan=8)
    x0 = o.variables
    rows.append({"electricity USD/kWh": price, **o.variables, "USD/t CH4": o.value,
                 "area m2": o.result.total_area, "power kW": o.result.total_power_kW,
                 "feasible": o.feasible})
pd.DataFrame(rows).set_index("electricity USD/kWh")
"""),
    md(r"""
## 4. Other objectives and your own layouts

* `objective="area"`, `"power"` or `"energy"` (kWh per tonne), or any function of the flowsheet
  result, e.g. `objective=lambda r: r.total_area + 5 * r.total_power_kW`.
* `product` takes a product label (`"slow-gas product"`, `"fast-gas product"`) or any stream name.
* The design function can build any layout (single stage, three-stage, a sweep on one stage) and
  expose any variable: pressures, stage cuts, flow patterns fixed inside it.

The optimiser finds a *local* optimum from the best scanned point. For an important design, check
`history` or repeat with a larger `n_scan` and a different `seed`.
"""),
]

# ---------------------------------------------------------------------------
NOTEBOOKS["10_sweep_and_shell_pressure_drop.ipynb"] = [
    md(r"""
# 10 · Permeate sweep gas and shell-side pressure drop

Two refinements of the module models:

1. **Sweep gas.** A gas fed into the permeate channel at its closed end dilutes the permeate,
   lowers the fast-gas partial pressure on the permeate side and so raises the driving force. It
   is used in post-combustion CO₂ capture (Merkel et al., 2010, sweep the second stage with the
   boiler's combustion air) and in dehydration (a dry sweep).
2. **Shell-side pressure drop.** In rating mode, the feed flowing along the fibre bundle loses
   pressure as well as the permeate inside the bores.
"""),
    code(SETUP),
    md(r"""
## 1. Sweep gas in a CO₂-capture stage

Flue gas, 13 % CO₂, Polaris membrane (Merkel 2010), 1.1 bar feed and 0.22 bar permeate, counter-
current flow. The stage cut keeps its meaning, the fraction of the **feed** that crosses the
membrane; the permeate outlet carries that gas plus the sweep.
"""),
    code("""
m  = Membrane(MEMBRANE_PRESETS["Polaris CO2/N2 (Merkel 2010)"])
xf = FEED_PRESETS["flue gas, 13% CO2 (Merkel 2010)"]
rows = []
for s in (0.0, 0.02, 0.05, 0.10, 0.20):
    r = solve_stage(m, xf, 0.12, 1.1, 0.22, "countercurrent", 1.0,
                    sweep_flow=s, sweep_composition={"N2": 1.0})
    rows.append({"sweep / feed": s, "CO2 recovery": r.recovery("CO2"),
                 "CO2 in retentate": r.x_ret[0], "CO2 in permeate": r.y_perm[0],
                 "area m2 per mol/s": r.area, "mass balance": r.mass_balance_error()})
pd.DataFrame(rows).set_index("sweep / feed")
"""),
    md(r"""
At the same stage cut, a small sweep raises the CO₂ recovery and cuts the area sharply, because
the permeate-side CO₂ partial pressure at the retentate end falls towards zero. The price is a
more dilute permeate: the sweep leaves with it. That is why Merkel's design sends the swept
permeate back to the boiler instead of to the CO₂ product.

The figure shows the same effect along the module.
"""),
    code("""
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
for k, s in enumerate((0.0, 0.05, 0.2)):
    r = solve_stage(m, xf, 0.12, 1.1, 0.22, "countercurrent", 1.0, profile=True,
                    sweep_flow=s, sweep_composition={"N2": 1.0})
    p = r.profile
    lab = f"sweep {s:g} × feed"
    ax[0].plot(p["area_fraction"], 100 * p["x"][0], color=gasmem_plot.SERIES[k], label=lab)
    ax[1].plot(p["area_fraction"], 1.1 * p["x"][0] - 0.22 * p["y"][0],
               color=gasmem_plot.SERIES[k], label=lab)
ax[0].set_ylabel("CO₂ on feed side (%)"); ax[1].set_ylabel("CO₂ partial-pressure difference (bar)")
for a in ax:
    a.set_xlabel("area fraction from the feed end"); a.legend()
plt.tight_layout()
"""),
    md(r"""
### An air sweep carries oxygen into the retentate

With combustion air as the sweep, oxygen is on the permeate side but not in the feed, so it
permeates **backwards** into the retentate. The model handles this; here the O₂ permeance is
taken as 2.5 × N₂, an assumption for illustration.
"""),
    code("""
m3 = Membrane({"CO2": 1000, "N2": 20, "O2": 50})
r = solve_stage(m3, {"CO2": 0.13, "N2": 0.87, "O2": 0.0}, 0.12, 1.1, 0.22, "countercurrent", 1.0,
                sweep_flow=0.1, sweep_composition={"N2": 0.79, "O2": 0.21})
print(r.summary())
print(f"mass-balance error {r.mass_balance_error():.1e}")
"""),
    md(r"""
### Check against an independent solver

The toolkit solves the counter-current problem by shooting from the retentate end. Here the same
problem is solved with SciPy's collocation boundary-value solver (`solve_bvp`), written from
scratch with the area as an unknown parameter.
"""),
    code("""
from scipy.integrate import solve_bvp
xF, S, z, th = np.array([0.13, 0.87]), 0.1, np.array([0.0, 1.0]), 0.12
ph, pl, Q = 1.1 * BAR, 0.22 * BAR, m.Q

def f(s, Y, p):              # s = area fraction from the feed end; p[0] = total area
    N, P = Y[:2], Y[2:]
    J = Q[:, None] * (ph * N / N.sum(0) - pl * P / P.sum(0))
    return np.vstack([-p[0] * J, -p[0] * J])

def bc(Ya, Yb, p):
    return np.array([Ya[0] - xF[0], Ya[1] - xF[1], Yb[2] - S * z[0] - 1e-30,
                     Yb[3] - S * z[1], Yb[0] + Yb[1] - (1 - th)])

s = np.linspace(0, 1, 50)
Y0 = np.vstack([np.outer(xF, 1 - th * s), np.outer([0.1, 0.12], 1 - s) + np.outer(S * z, s)])
bvp = solve_bvp(f, bc, s, Y0, p=[50.0], tol=1e-8, max_nodes=100000)
r = solve_stage(m, xF, th, 1.1, 0.22, "countercurrent", 1.0, sweep_flow=S,
                sweep_composition={"N2": 1.0})
pd.DataFrame({"gasmem (shooting)": [r.area, r.x_ret[0]],
              "solve_bvp (collocation)": [bvp.p[0], bvp.y[0, -1] / bvp.y[:2, -1].sum()]},
             index=["area m2 per mol/s", "CO2 in retentate"])
"""),
    md(r"""
Sweeps also work in flowsheets: give a stage `StageSpec(..., sweep_ratio=0.1,
sweep_composition={"N2": 0.79, "O2": 0.21})`. The sweep then appears as an extra input stream
(`sweep 2` for stage 2) and is included in the flowsheet mass balance.

## 2. Shell-side pressure drop in a fixed module

Give the module a **packing fraction** φ (fibre cross-section over shell cross-section, typically
0.4–0.6). The feed then flows along the fibres through the free shell area, treated as laminar flow
in a channel of hydraulic diameter d<sub>h</sub> = d<sub>o</sub>(1−φ)/φ. Real bundles also lose
pressure at the ports and baffles; `shell_friction_factor` scales the coefficient to match a
measured pressure drop.

The case is biogas in a 1 744 m² module (the single-stage design of the README), fed at 1 000 and
2 000 Nm³/h.
"""),
    code("""
mb  = Membrane(MEMBRANE_PRESETS["Polyimide CO2/CH4, a=35 (Makaruk 2010)"])
xfb = FEED_PRESETS["biogas, 60% CH4 (Makaruk 2010)"]
rows = []
for flow in (1000, 2000):
    for phi, fric in ((None, 1.0), (0.5, 1.0), (0.6, 1.0), (0.6, 30.0)):
        mod = HollowFibreModule.for_area(1744, length=1.0, d_inner=200e-6, d_outer=400e-6,
                                         packing_fraction=phi, shell_friction_factor=fric)
        r = rate_module(mb, xfb, nm3h_to_mols(flow), 10, 1, mod, "countercurrent", profile=False)
        rows.append({"feed Nm3/h": flow, "packing": phi or "none", "friction factor": fric,
                     "shell dp bar": r.extra["shell_dp_bar"], "bore dp bar": r.extra["bore_dp_bar"],
                     "stage cut": r.theta, "CH4 retentate": r.x_ret[1]})
pd.DataFrame(rows).set_index(["feed Nm3/h", "packing", "friction factor"])
"""),
    md(r"""
With the bundle alone, the shell-side loss is a few millibar: the feed velocity in the shell is low
and the free area large. It matters when the measured module pressure drop is much larger
(ports, baffles, fouling), or at low feed pressure, where every 100 mbar is a large part of the
driving force. The profile below shows both pressures along the module for the high-friction case.
"""),
    code("""
mod = HollowFibreModule.for_area(1744, packing_fraction=0.6, shell_friction_factor=30.0)
r = rate_module(mb, xfb, nm3h_to_mols(2000), 10, 1, mod, "countercurrent")
p = r.profile
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
ax[0].plot(p["area_fraction"], p["p_feed_bar"], color=gasmem_plot.SERIES[0])
ax[0].set_ylabel("feed (shell) pressure, bar")
ax[1].plot(p["area_fraction"], p["p_perm_bar"], color=gasmem_plot.SERIES[1])
ax[1].set_ylabel("permeate (bore) pressure, bar")
for a in ax:
    a.set_xlabel("length fraction from the feed end")
plt.tight_layout()
print(mod)
"""),
]

# ---------------------------------------------------------------------------
NOTEBOOKS["11_your_own_data.ipynb"] = [
    md(r"""
# 11 · Compare the models with your own data

If you have measurements from a module (laboratory, pilot or plant), you can check the models
against them in a few lines, and fit permeances to them.

1. Get a blank data sheet with `template`, fill it in (one row per operating point).
2. `compare` runs the model at every point and pairs each prediction with its measurement.
3. `fit_permeances` finds the permeances that best reproduce your data.

**Please share what you find.** Comparisons with real modules are the most useful contribution to
this toolkit, whether the agreement is good or poor. See *Sharing your comparison* at the end.
"""),
    code(SETUP),
    code("""
from gasmem.userdata import template, read, compare, fit_permeances
template(["O2", "N2"])
"""),
    md(r"""
| Column | Meaning |
|---|---|
| `point` | your label for the operating point |
| `pattern` | optional per row; otherwise the `pattern` argument |
| `p_feed_bar`, `p_perm_bar` | absolute pressures, bar |
| `feed_flow_mol_s` (or `feed_flow_Nm3h`) | needed to compare areas, or to rate a module |
| `x_feed_<comp>` | feed mole fractions |
| `stage_cut` | measured stage cut: the model runs at it (**design mode**) |
| `area_m2` | membrane area: compared in design mode, or used to predict the stage cut (**rating mode**) |
| `y_perm_<comp>`, `x_ret_<comp>` | measured permeate and retentate mole fractions |

Use fractions, not percent, and leave cells empty when a quantity was not measured.

## Worked example

As an example, the measured points of Feng et al. (1999), Figs. 4–5 (air separation, bore-side
feed, counter-current, 7.83 bar feed, 0.93 bar permeate) are put into the sheet format. These are
digitised values shipped with the toolkit; replace them with your own file, e.g.
`data = read("my_module.csv")`.
"""),
    code("""
from gasmem.datasets import load
feng = load("Feng1999").query("figure in ['Fig. 4', 'Fig. 5'] and "
                              "series == 'bore-side feed, countercurrent'")
rows = []
for _, p in feng.iterrows():
    row = {"point": f"{p.figure}, θ={p.x:.3f}", "p_feed_bar": 7.828, "p_perm_bar": 0.933,
           "x_feed_O2": 0.205, "x_feed_N2": 0.795, "stage_cut": p.x}
    row["x_ret_N2" if "N2" in p.quantity else "y_perm_O2"] = p.y / 100
    rows.append(row)
data = pd.DataFrame(rows)
data.to_csv("example_user_data.csv", index=False)      # this is what your file would look like
read("example_user_data.csv").head()
"""),
    md("## Compare"),
    code("""
m = Membrane(MEMBRANE_PRESETS["Asymmetric hollow fibre air (Feng 1999)"])
cmp = compare(m, "example_user_data.csv", pattern="countercurrent")
display(cmp.summary())
cmp.plot();
"""),
    md(r"""
`cmp.table` holds every pair (point, quantity, measured, predicted, error) for your own plots, and
`cmp.failed` lists any point the model could not run, with the reason.

## Fit permeances

Compositions depend only on permeance *ratios*, so without area data the fit returns the
selectivity, with the fastest gas held at its starting value. Here the fit starts from a poor guess,
α = 3, to show that the data, not the preset, fix the answer.
"""),
    code("""
start = Membrane({"O2": 9.2, "N2": 9.2 / 3})
fitted, cmp_fit, info = fit_permeances(start, "example_user_data.csv", pattern="countercurrent")
print(fitted)
print("fitted O2/N2 selectivity:", round(fitted.selectivity("O2", "N2"), 3),
      " (preset from Feng's paper:", round(m.selectivity("O2", "N2"), 3), ")")
print(info["note"])
print("95 % range:", info.get("permeance_95pct_range_GPU"))
cmp_fit.summary()
"""),
    md(r"""
If your sheet has areas and feed flows, both permeances are fitted, and the result is the effective
permeance of your module (which includes any effects the model leaves out). Try other flow patterns
too: the pattern that fits best tells you how your module behaves (Pan 1986 found cross-flow for
high-flux asymmetric fibres).

## Sharing your comparison

Open an issue on GitHub (https://github.com/SinaGhanbarii/gasmem/issues) with:

* the data sheet (CSV), or the summary table if the data are confidential;
* the membrane (or permeances), module type and flow pattern you assumed;
* `cmp.summary()` and the parity plot;
* where the data come from (paper, own measurements) and whether it may be added to the toolkit's
  validation set, with credit.

Poor agreement is just as useful as good agreement: it shows where the models need work.
"""),
]


def main(run=False, only=None):
    NB.mkdir(exist_ok=True)
    for name, cells in NOTEBOOKS.items():
        if only and not any(name.startswith(o) for o in only):
            continue
        nb = nbf.v4.new_notebook()
        nb.cells = [md(BADGE.format(name=name))] + cells
        nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3",
                                     "language": "python"}
        path = NB / name
        nbf.write(nb, path)
        print("wrote", path.relative_to(ROOT))
    if run:
        from nbconvert.preprocessors import ExecutePreprocessor
        for name in NOTEBOOKS:
            if only and not any(name.startswith(o) for o in only):
                continue
            path = NB / name
            nb = nbf.read(path, as_version=4)
            ExecutePreprocessor(timeout=900, kernel_name="python3").preprocess(
                nb, {"metadata": {"path": str(NB)}})
            nbf.write(nb, path)
            print("executed", name)


if __name__ == "__main__":
    # python tools/build_notebooks.py [--run] [08 07 ...]   (default: all notebooks)
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    main(run="--run" in sys.argv, only=args or None)
