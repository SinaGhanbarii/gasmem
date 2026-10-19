# Contributing to gasmem

Thank you for helping. The most valuable contribution is **a comparison of the models with real
data**: your own measurements on a laboratory, pilot or industrial module, or data from a paper.
Bug reports, corrections and code are welcome too.

## Share a comparison with your data

1. Put your data in the sheet format (notebook `11_your_own_data.ipynb` explains every column):

   ```python
   from gasmem.userdata import template, compare, fit_permeances
   template(["CO2", "CH4"], "my_module.csv")          # fill this in
   ```

2. Run the comparison and, if you like, fit permeances:

   ```python
   from gasmem import Membrane
   m = Membrane({"CO2": 100, "CH4": 100 / 35})       # your best estimate, in GPU
   c = compare(m, "my_module.csv", pattern="countercurrent")
   print(c.summary()); c.plot()
   ```

3. Open an issue with the **"Share validation data"** form
   (https://github.com/SinaGhanbarii/gasmem/issues/new/choose) and attach:
   - the data sheet (CSV), or only `c.summary()` and the parity plot if the data are confidential;
   - the membrane or permeances, the module type and the flow pattern you assumed;
   - where the data come from (your measurements, or the paper with its DOI and figure/table);
   - whether the data may be added to the toolkit's validation set (you will be credited).

Poor agreement is as useful as good agreement: it shows where the models need work. Data you share
are added only with your permission and under the repository's MIT licence; please share only data
you have the right to share.

## Report a bug

Open an issue with a short script that reproduces the problem, the full error message, and the
output of `python -c "import gasmem, numpy, scipy; print(gasmem.__version__, numpy.__version__,
scipy.__version__)"`.

## Contribute code

1. Fork the repository and create a branch.
2. Install for development: `python -m pip install -e ".[notebooks,test]"`.
3. Add a test in `tests/` for every change in behaviour; models should be checked against an
   analytical solution, an independent solver or published data.
4. The notebooks are generated from `tools/build_notebooks.py`: edit the builder, then run
   `python tools/build_notebooks.py --run 09` (with the notebook number) to rebuild and execute.
5. Run `python -m pytest -q`, add a line to `CHANGELOG.md`, and open a pull request. The tests and
   all notebooks run automatically on every pull request.

New membrane presets or datasets need a reference (DOI) and, for digitised data, a record of how the
values were read (see `tools/digitised_data.py`).
