# Changelog

All notable changes to `gasmem` are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[semantic versioning](https://semver.org/): a change in the first number may break existing
scripts, a change in the second adds features, a change in the third fixes bugs.

## [1.2.0] - 2026-10-19

### Added
- **Permeate sweep gas** for co-current and counter-current stages: `solve_stage(...,
  sweep_flow=..., sweep_composition={...})`, and `StageSpec(..., sweep_ratio=...,
  sweep_composition=...)` in flowsheets. Sweep components absent from the feed may permeate
  backwards into the retentate. Checked against an independent collocation solver (`solve_bvp`).
- **Shell-side (feed) pressure drop** in rating mode for cross-flow, co-current and counter-current
  modules: give `HollowFibreModule` a `packing_fraction` (and optionally a
  `shell_friction_factor`). Results report `shell_dp_bar` and the feed-pressure profile. New
  `PressureDropError` for feed flows that a module's pressure drops cannot pass.
- **Design optimisation** (`gasmem.optimise`): `stage_cut_for_spec` finds the stage cut for a
  purity or recovery specification; `optimise` chooses stage cuts, pressures or any other variable
  of any layout to minimise cost per tonne, area, power, specific energy or a custom objective,
  subject to purity and recovery targets.
- **Own-data tools** (`gasmem.userdata`): `template` (blank data sheet), `compare` (model versus
  measurements, summary statistics and parity plot) and `fit_permeances` (least-squares permeances
  with approximate 95 % ranges).
- Notebooks 09 (design optimisation), 10 (sweep and shell-side pressure drop) and 11 (your own data).
- "Open in Colab" badge on every notebook; in Colab the notebooks now download the toolkit from
  GitHub themselves.
- GitHub Actions workflow running the tests (Windows and Linux, Python 3.10 and 3.12) and every
  notebook; `CONTRIBUTING.md`; this changelog.

### Changed
- `StageResult.permeate_flow` includes the sweep flow, and permeate recovery with a sweep is the net
  transfer out of the feed. Without a sweep all results are unchanged.
- `StageResult.as_dict()` lists the recovery columns after the composition columns.

### Fixed
- Counter-current multicomponent shooting is more robust: compositions are normalised without
  overflow, and feed components present only at trace level (below 1e-14) are matched in absolute
  rather than relative terms.

## [1.1.0]

First version published on GitHub.

### Added
- Literature-based membrane and feed presets with sources (`MEMBRANE_INFO`, `FEED_INFO`,
  `describe`).
- Digitised validation data from Pan (1986), Feng et al. (1999) and Makaruk et al. (2010)
  (`gasmem.datasets`), and notebook 08 comparing the models with them point by point.
- Literature validation notebook 07 and regression tests against the published results.

## [1.0.0]

First version: well-mixed, cross-flow, co-current and counter-current stage models;
fixed-module rating with bore-side pressure drop; single-, two- and three-stage flowsheets with
recycle; compression energy and indicative cost; Robeson upper bounds; notebooks 00–06.
