## datasets.load_us_inflation()


Load quarterly US inflation for CPI, core CPI, PCE and core PCE.


Usage

``` python
datasets.load_us_inflation()
```


The bundled CSV holds quarterly averages of the monthly, seasonally adjusted price indexes `CPIAUCSL` (CPI, all items), `CPILFESL` (CPI less food and energy), `PCEPI` (PCE price index) and `PCEPILFE` (PCE less food and energy) from FRED, downloaded on 2026-11-08 by `scripts/build_us_inflation_csv.py`. The levels run from 1959Q1 to 2026Q3, the last quarter with three published months for every series. They are revised data as published at that date, not real-time vintages. The BLS published no CPI for October 2025 (the federal government shutdown of 2025), so that month of the two CPI series is the geometric mean of September and November 2025 before the quarterly average. The loader turns the levels into annualized quarterly log rates in percent, 400 \\ (\log P_t - \log P\_{t-1}), so the rates start in 1959Q2. All four series are public domain (FRED tag *Public Domain: Citation Requested*; sources BLS and BEA).


## Returns


`USInflation`  
The rates `(quarters, measures)` as `float32`, the quarter dates and the measure names, core PCE last.
