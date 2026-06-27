# Energy Demand Prediction

Predicts two operational energy outcomes from weather and calendar signals:
expected consumption in kWh and high-demand alert probability.

## Key Artifacts

- `energy_demand_slides.pdf` - concise presentation.
- `energy_demand_results.json` - final prediction artifact.
- `energy_demand_summary.txt` - result summary.
- `energy_demand_code.py` and `energy_demand_slides_code.py` - reproducible
  analysis and presentation code.
- `energy_demand_train.csv` and `energy_demand_test.csv` - project data.

## Validation Work

The `experiments/` folder contains exploratory plots, knot-sensitivity checks,
nested cross-validation benchmarks, out-of-fold diagnostics, Monte Carlo
stability checks, and test scripts used to validate the final configuration.
