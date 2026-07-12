# Examples


<a href="../../docs/examples/electricity_forecast.html" class="section-card" style="display: block; padding: 1.25rem 1.5rem; border: 1px solid #dee2e6; border-radius: 0.5rem; color: inherit; text-decoration: none;"><img src="thumbnails/electricity_forecast.png" class="section-card-img" style="width: 100%; border-radius: 0.375rem; margin-bottom: 0.75rem;" /></a>


Electricity demand forecasting


Forecast hourly electricity demand in Victoria, Australia with a varying-coefficient temperature effect built from a Hilbert space Gaussian process, hour-of-day and day-of-week seasonality, and a Student-t likelihood fit with SVI.


<a href="../../docs/examples/electricity_forecast_calibration.html" class="section-card" style="display: block; padding: 1.25rem 1.5rem; border: 1px solid #dee2e6; border-radius: 0.5rem; color: inherit; text-decoration: none;"><img src="thumbnails/electricity_forecast_calibration.png" class="section-card-img" style="width: 100%; border-radius: 0.375rem; margin-bottom: 0.75rem;" /></a>


Electricity demand forecasting: prior calibration


Calibrate the priors of the electricity demand model against domain knowledge about the temperature effect, iterating with prior predictive checks before refitting.


<a href="../../docs/examples/exponential_smoothing_state_space.html" class="section-card" style="display: block; padding: 1.25rem 1.5rem; border: 1px solid #dee2e6; border-radius: 0.5rem; color: inherit; text-decoration: none;"><img src="thumbnails/exponential_smoothing_state_space.png" class="section-card-img" style="width: 100%; border-radius: 0.375rem; margin-bottom: 0.75rem;" /></a>


Exponential Smoothing in State Space Form


Write seasonal damped-trend exponential smoothing in innovations state space form so forecast uncertainty is propagated correctly, and fit it with Hamiltonian Monte Carlo.


<a href="../../docs/examples/forecasting_univariate.html" class="section-card" style="display: block; padding: 1.25rem 1.5rem; border: 1px solid #dee2e6; border-radius: 0.5rem; color: inherit; text-decoration: none;"><img src="thumbnails/forecasting_univariate.png" class="section-card-img" style="width: 100%; border-radius: 0.375rem; margin-bottom: 0.75rem;" /></a>


Univariate forecasting


Forecast weekly BART ridership with a random-walk local level, Fourier seasonality, and a Student-t likelihood fit with SVI, then evaluate the model with rolling-origin backtesting.


<a href="../../docs/examples/fresh_retail_stockout.html" class="section-card" style="display: block; padding: 1.25rem 1.5rem; border: 1px solid #dee2e6; border-radius: 0.5rem; color: inherit; text-decoration: none;"><img src="thumbnails/fresh_retail_stockout.png" class="section-card-img" style="width: 100%; border-radius: 0.375rem; margin-bottom: 0.75rem;" /></a>


Forecasting retail demand under stockouts


Forecast 1,000 daily store-product demand series from FreshRetailNet-50K with a damped-trend panel model, store-pooled promotion effects, a launch indicator, and a floored saturating availability factor that handles noisy stockout labels, fit with SVI and a custom optax optimizer, ending with a full-availability counterfactual forecast of uncensored demand for planning.


<a href="../../docs/examples/hierarchical_forecasting_1.html" class="section-card" style="display: block; padding: 1.25rem 1.5rem; border: 1px solid #dee2e6; border-radius: 0.5rem; color: inherit; text-decoration: none;"><img src="thumbnails/hierarchical_forecasting_1.png" class="section-card-img" style="width: 100%; border-radius: 0.375rem; margin-bottom: 0.75rem;" /></a>


Hierarchical forecasting I


Forecast hourly BART arrivals from eight origin stations to a fixed destination with a hierarchical model that pools information across series, fit with SVI.


<a href="../../docs/examples/hierarchical_forecasting_2.html" class="section-card" style="display: block; padding: 1.25rem 1.5rem; border: 1px solid #dee2e6; border-radius: 0.5rem; color: inherit; text-decoration: none;"><img src="thumbnails/hierarchical_forecasting_2.png" class="section-card-img" style="width: 100%; border-radius: 0.375rem; margin-bottom: 0.75rem;" /></a>


Hierarchical forecasting II


Scale the hierarchical model to the full 50 by 50 origin-destination panel of hourly BART ridership, forecasting all 2,500 series jointly with SVI.


<a href="../../docs/examples/inference_methods_comparison.html" class="section-card" style="display: block; padding: 1.25rem 1.5rem; border: 1px solid #dee2e6; border-radius: 0.5rem; color: inherit; text-decoration: none;"><img src="thumbnails/inference_methods_comparison.png" class="section-card-img" style="width: 100%; border-radius: 0.375rem; margin-bottom: 0.75rem;" /></a>


Comparing inference methods: NUTS, SVI, Pathfinder, and MCLMC


Fit the same weekly BART ridership model with NUTS, SVI, Pathfinder, and MCLMC without touching the model code, and compare their forecasts with CRPS.
