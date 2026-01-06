# Discount Recommendation Engine (Task 1)

This project builds a discount optimization system that learns historical rebate behavior, predicts rebates, applies business guardrails, and outputs safe, explainable discount recommendations.

## Structure

```
iffco_discount_model/
├── data/
│   └── contracts.csv            # Place your dataset here (50k+ rows)
├── outputs/
│   └── rebate_recommendations.csv
├── models/
│   └── discount_model.pkl
└── discount_model.ipynb         # Optional analysis notebook
```

## Quick Start

1. Place your dataset at `iffco_discount_model/data/contracts.xls` (or `.xlsx`). Optionally set `--sheet-name`.
2. Install deps:

```
pip install -r iffco_discount_model/requirements.txt
```

3. Run the pipeline:

```
python iffco_discount_model/discount_pipeline.py --data-path iffco_discount_model/data/contracts.xls --sheet-name Sheet1
```

Outputs are saved to `iffco_discount_model/outputs/` and the model to `iffco_discount_model/models/`.

## Notes for Large Data
- Uses `OneHotEncoder(handle_unknown='ignore', sparse=True)` to keep memory usage low.
- RandomForestClassifier with constrained depth and leaf size for robustness.
- Reads Excel via pandas (`.xls` requires `xlrd<2.0`, `.xlsx` uses `openpyxl`).
- Date parsing uses `pd.to_datetime(..., errors='coerce', dayfirst=True)` with engineered date features.

## Business Guardrails
- GT channel capped at 10%.
- LOC_MT + Ice creams capped at 15%.
- Bakery capped at 12.5%.

## CLI Options
```
python iffco_discount_model/discount_pipeline.py --help
```
- `--data-path`: path to contracts CSV
- `--outputs-dir`: output directory (default: `iffco_discount_model/outputs`)
- `--models-dir`: models directory (default: `iffco_discount_model/models`)
- `--check-only`: validate dataset & required columns without training
 - `--sheet-name`: Excel sheet name (default: first sheet)
