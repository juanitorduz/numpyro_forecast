# M5 notebook 2 (hierarchical count state space model): work in progress

Temporary notes to resume the work on `docs/examples/m5_hierarchical_state_space.py` (jupytext percent source; the executed `.ipynb` is not committed yet). Delete this file before the PR is ready.

## State

- The source runs end to end in a fast validation (150 SVI steps, 20 draws) with the beartype runtime checker active. The full execution (`uv run jupytext --to notebook --execute docs/examples/m5_hierarchical_state_space.py`, about 25 to 40 minutes on the M4 Pro) was started but its output was not captured before the session ended: re-run it.
- Prose cells are placeholders (`INTRO_PLACEHOLDER`, `DATA_PLACEHOLDER`, ..., `DISCUSSION_PLACEHOLDER`). Result-independent drafts for the first eight placeholders are in `/tmp/nb2_prose_static.py` of the session machine and reproduced below; the result-dependent ones (ladder, seed noise, final choice, final fit and posterior, items, holdout, held-out window, comparison, timing, discussion) must be written from the executed outputs.
- The final model is selected automatically in the notebook: the simplest ladder rung within one seed-noise of the best rung on the selection window (origin day 1,878). The held-out window (origin 1,913) and the evaluation window are reported afterwards.
- Package: `numpyro_forecast.datasets.load_m5()` (already in PR #144) is the loader.

## Pilot findings that shaped the design (origin day 1,878, WS-CRPS mean over 12 levels; top-down baseline on that window: 0.5585, levels 1-9: 0.490, levels 10-12: 0.766)

- Joint model with daily store and store-department random walks and daily shocks shared by all items (minibatch SVI, DCT reparam): 0.71 to 0.87, biased totals (+17% to -9%), walk scales 0.03 to 0.08 per day. Daily shared latents absorb noise as level and over-disperse the horizon at levels 3 to 9.
- Two-stage model (full-batch NB state space on the 70 store-department series, items rescaled to it): level 1 improves (0.305) but levels 3 and 9 degrade (0.67 to 0.75), same walk-scale problem (0.03 per day even under a HalfNormal(0.005) prior, DCT or Haar). AR(1) level reverts to the mean and biases the total by -10%.
- Item NB model with the intercept at the last-28-day level and an end-anchored 28-day block walk: 0.5285 (5 years, 600-series minibatch, `init_to_value` at the recent level) and 0.5371 (2 years, 1,500-series minibatch, intercept prior centered on the recent level); items 0.78 to 0.82, levels 1-9 0.43 to 0.46, but level-1 coverage 0.3 to 0.5 (bands too narrow).
- Per-item price elasticities are unidentified (posterior sd 1.6 to 3) and produced single draws of 1.7 million units: pooled at the department level, and the log mean is clipped at log(2,000).
- Adding end-anchored store and group block walks plus weekly zero-sum store shocks to the item model: 0.5607 on 2 years (level-1 coverage 0.71 / 0.18, levels 1-9 0.487, items 0.783): more calibrated but not better in CRPS on this window. The ladder measures it.
- `jax.random.poisson` on a (300, 28, 30490) array takes 266 s and 10 GB on this CPU; NumPy takes 6.7 s. The 30,490-series forecast therefore reads the horizon mean and dispersion from a forecast-only model instance and samples in NumPy.

## Next steps

1. Run the notebook fully; read the ladder table, the seed noise, the selected rung, the holdout table, the held-out window table and the figures.
2. Write the prose for every placeholder from the outputs (no em-dashes, `$94\%$ HDI`, one line per paragraph), keep the code unchanged, re-execute if any code changes.
3. Checks: `uv run ruff check` and `ruff format --check` on the `.ipynb`, `uv run ty check`, `prek run`, `pytest tests/test_docstring_markup.py tests/test_build_docs.py tests/test_docs_reference.py`; README entry under "Hierarchical and panel"; delete the `.py` and this file; commit, push, open the PR (stacked on `m5-starter-kit-notebook`, retarget to `main` after #144 merges).

## Draft prose (result-independent)

### INTRO_PLACEHOLDER

This notebook builds on the [M5 baselines](m5_forecasting.html), the port of the three [Pyro M5 Starter Kit](https://github.com/pyro-ppl/Pyro-M5-Starter-Kit) models, and asks what the full NumPyro toolbox adds on the same data: the 30,490 daily unit-sales series of the M5 competition (3,049 items in 10 Walmart stores of 3 states), a 28-day horizon, and the weighted scaled CRPS over the 42,840 aggregates of the 12-level hierarchy that the baselines are scored with. On the evaluation window the kit's top-down model scored a WS-CRPS of 0.569 (mean over levels), the middle-out model 0.634 and the bottom-up model 0.728; the top-down model is the one to beat.

The candidate is a single hierarchical model of every item in every store: a negative binomial likelihood with item-level dispersion, an intercept centered on each series' level at the forecast origin, an end-anchored random walk of the item level over 28-day blocks (a local level in state space form), zero-sum weekday effects shared by the items of a store-department with zero-sum item deviations, SNAP and calendar-event effects, yearly seasonality and a price elasticity, all partially pooled, and trained with minibatch SVI on 1,500 of the 30,490 series per step. The model is built up in a **ladder** of six variants so the value of each component can be read off the score, on a selection window that never touches the windows the final model is reported on.

> **How the ladder was chosen.** Two design rounds happened before this notebook: a joint model with daily store and department random walks and daily shocks shared by all items, and a two-stage model with a full-batch state space model of the 70 store-department series. Both lost to the top-down baseline on the selection window, for the same reason: daily latent processes fitted by mean-field SVI absorb day-to-day variation as level changes and project it as a random walk over the horizon, which widens the bands at the store and department levels by a factor the data does not support. The ladder below keeps the shared state space components at a coarser resolution (28-day blocks and zero-sum weeks) so that the score decides.

### DATA_PLACEHOLDER

`load_m5()` downloads the [Nixtla mirror](https://github.com/Nixtla/m5-forecasts) of the competition files once and reads them: the daily sales over the training period (`d_1` to `d_1941`) followed by the 28 evaluation days (`d_1942` to `d_1969`), the weekly shelf prices repeated over the days of every series (`NaN` when the item was not on the shelf), the series identifiers, the calendar and the official evaluation weights. The calendar carries two event columns; the second one is rare, but one of its five days (Father's Day 2016) falls inside the evaluation window, so both are indexed.

### HIERARCHY_PLACEHOLDER

The competition scores 42,840 series: the 30,490 items in stores (level 12) and their sums over the 11 coarser groupings. A label and a dense group id per level turn into a sparse `(30,490, 42,840)` summation matrix, so any array of bottom-level values (data or forecast draws) is aggregated to every level with one sparse product. The item, store, department and store-department ids of every series index the hierarchical parameters of the model.

### SCORING_PLACEHOLDER

Same definition as the baselines notebook (WS-CRPS formula), plus the means over levels 1 to 9 and 10 to 12 reported separately because the two halves reward different things: calibrated common uncertainty at the top, a sharp level per item at the bottom. WSPL on the evaluation window.

### FEATURES_PLACEHOLDER

Three per-series channels (day index, `saled`, log price ratio to the item's mean listed price over the first 1,843 days); sixty nonzero sales on unlisted or closed days set to zero in the training counts; calendar tables (weekday, SNAP by state, two event indices, four yearly harmonics) looked up inside the model.

### MODEL_PLACEHOLDER

The model equation and the component list (level at the origin, item block walk, shared block walks and weekly shocks, weekday, calendar, price, likelihood), the minibatch mechanics (`create_plates`, `numpyro.subsample`, explicit block and week plates because `AutoNormal` keeps one size per plate name, no `time_reparam` at this resolution), and the forecast-only instance used for the 30,490-series forecast with NumPy sampling, `forecast()` on the focus items.

### FIT_PLACEHOLDER

`AutoNormal` with median initialization, one-cycle Adam peaking at 0.03 with clipped gradients, 3,000 steps over the last two years (about 150 visits per series with 1,500-series minibatches); the `backtest` closure builds the counts and the origin levels from the window, fits, and returns bottom-level draws 50 posterior draws at a time.

### BASELINE_PLACEHOLDER

The kit's top-down model re-run in the same harness (same windows, same draws); the other two baselines quoted from the baselines notebook.
