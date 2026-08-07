"""Extract paper figures from the stored PNG outputs of the example notebooks.

Figures are addressed by (notebook, PNG ordinal): the ordinal counts cells with an
``image/png`` output from the top of the notebook. Ordinals are stable as long as the
notebooks are not re-executed; the EXPECTED_PNG_COUNTS check fails loudly on drift.
Each manifest entry records the first source line of its cell so re-runs are auditable.
"""

import base64
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = REPO_ROOT / "docs" / "examples"
FIGURES = Path(__file__).resolve().parents[1] / "figures"

EXPECTED_PNG_COUNTS = {
    "availability_tsb.ipynb": 14,
    "censored_demand.ipynb": 5,
    "fresh_retail_stockout 2.ipynb": 22,
}

# (notebook, png ordinal) -> output filename. Cell provenance in the comment above each entry.
MANIFEST = {
    # cell 31: avail_train_np = np.asarray(available_train)
    ("availability_tsb.ipynb", 7): "fig02_gated_vs_counterfactual_probability.png",
    # cell 47: p_true = 1 - np.exp(-lam)
    ("availability_tsb.ipynb", 12): "fig03_probability_recovery_scatter.png",
    # cell 36: trailing_run = np.cumprod(...)
    ("availability_tsb.ipynb", 9): "fig04_forecast_opens_at_demand_level.png",
    # cell 6: fig, (ax_top, ax_bot) = plt.subplots(
    ("censored_demand.ipynb", 0): "fig05_censored_dgp_anatomy.png",
    # cell 23: forecast_naive_pp = stacked_draws(...)
    ("censored_demand.ipynb", 4): "fig06_censored_vs_plain_forecasts.png",
    # cell 17: def hdi_label(...)
    ("censored_demand.ipynb", 2): "fig07_in_sample_fit_above_cap.png",
    # cell 83: posterior_factor = (
    ("fresh_retail_stockout 2.ipynb", 19): "fig08_posterior_availability_factor.png",
    # cell 79: comparison_id = "22::267"
    ("fresh_retail_stockout 2.ipynb", 17): "fig09_sales_vs_demand_series.png",
}


def extract() -> None:
    """Decode the manifest's PNG outputs into ``paper/figures/``."""
    FIGURES.mkdir(exist_ok=True)
    for nb_name, expected in EXPECTED_PNG_COUNTS.items():
        nb = json.loads((EXAMPLES / nb_name).read_text())
        pngs = []
        for cell in nb["cells"]:
            for output in cell.get("outputs", []):
                data = output.get("data") or {}
                if "image/png" in data:
                    pngs.append(data["image/png"])
                    break
        if len(pngs) != expected:
            sys.exit(
                f"{nb_name}: found {len(pngs)} PNG outputs, expected {expected}. "
                "The notebook changed; re-audit the manifest ordinals."
            )
        for (manifest_nb, ordinal), out_name in MANIFEST.items():
            if manifest_nb == nb_name:
                (FIGURES / out_name).write_bytes(base64.b64decode(pngs[ordinal]))
                print(f"{out_name} <- {nb_name}[{ordinal}]")


if __name__ == "__main__":
    extract()
