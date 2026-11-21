# Microplastic CCC Stacking

This repository contains a literature dataset and a stacking workflow for predicting the critical coagulation concentration (CCC) of microplastic and nanoplastic particles.

## Files

| File | Purpose |
| --- | --- |
| `Data.xlsx` | English-language workbook with 507 observations from 45 source studies |
| `train_stacking.py` | Preprocessing, model fitting, stacking, and grouped cross-validation |
| `requirements.txt` | Python dependencies pinned to the reproduction environment |

## Run the evaluation

Use Python 3.12 and run the commands from the repository root.

Windows PowerShell

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe train_stacking.py --jobs 2
```

Linux or macOS

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 .venv/bin/python train_stacking.py --jobs 2
```

The script reads `Data.xlsx` next to the script by default. Use `--data PATH` to select another workbook or `--sheet NAME` to select another worksheet. The console reports R² for the five standalone models, followed by Stacking. All reported R² values use `log10(CCC in mM)`.

### Run in Jupyter

Place the notebook and the four release files in the same directory. Use a Python 3.12 kernel. Run the first line in a code cell if the dependencies need to be installed, and then run the second line in a new cell.

```ipython
%pip install -r requirements.txt
%run train_stacking.py --jobs 2
```

Running the script contents directly in a notebook cell also works. In that case the script looks for `Data.xlsx` in the kernel's working directory. Use `%pwd` to check that directory. The English workbook has a sheet named `Standardized data`.

## Dataset

The sheet is named `Standardized data`. Its 16 populated columns are in the original order. Predictor labels follow the manuscript figure. Units follow the source workbook.

| Column | Role | Unit or interpretation |
| --- | --- | --- |
| CCC (mM) | Target | mM |
| Cation type | Categorical predictor | Cation identity with valence as recorded, for example `Na(1)` |
| Anion type | Categorical predictor | Anion identity with valence as recorded, for example `Cl(1)` |
| Temperature | Numeric predictor | °C |
| Initial Zeta potential | Numeric predictor | mV |
| Particle type | Categorical predictor | `NPs` or `MPs` |
| pH | Numeric predictor | Dimensionless |
| Plastic surface modification type | Categorical predictor | `Not reported` retains the original reporting category |
| Plastic type | Categorical predictor | Polymer label |
| Plastic concentration | Numeric predictor | mg/L |
| Initial particle size | Numeric predictor | nm |
| Coexisting species type | Categorical predictor | `Absent` denotes the recorded absence of a coexisting species |
| Coexisting species concentration | Numeric predictor | mg/L |
| Irradiation time | Numeric predictor | h |
| Literature title | Source metadata | Title of the original study |
| Literature ID | Group metadata | Study identifier used for grouped evaluation |

The workbook retains numerical values, blank cells, and source titles. During training, the two translated category labels are mapped to their original internal tokens to preserve the ordering of one-hot encoded columns. This preserves the feature indices seen by models that subsample features. The model uses the 13 predictors in workbook order.

## Models and evaluation

The five standalone models are XGBoost (XGB), Extra Trees (EXT), gradient boosting regression (GBR), Random Forest (RF), and LightGBM (LGBM). Stacking reuses their predictions and adds an internal GBR fit as an anchor. The standalone and anchor GBR fits use 150 and 750 trees, respectively.

Grouped inner folds within each outer training partition are used to select preprocessing and generate first-level out-of-fold (OOF) predictions. The stacking stages use grouped resampling of these predictions to select blending and calibration settings. The reported R² values use the held-out outer predictions.

Groups are connected components formed by linking records with the same study ID or identical predictor signatures. The five-fold outer protocol assigns whole groups to folds using the observed log10(CCC) distribution over the full dataset. After assignment, each outer test fold is held out from preprocessing, model fitting, and stacking selection. This reproduces the original target-balanced evaluation.

Pooled OOF R² uses all outer predictions at once. The fold-mean R² averages individual fold scores.

## Reference results

The R² values below were reproduced on Linux x86-64 with Python 3.12, the pinned dependencies, and two workers.

| Model | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Fold mean | Pooled OOF |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| XGB | 0.7885 | 0.8648 | 0.6482 | 0.7918 | 0.7907 | 0.7768 | 0.7899 |
| EXT | 0.7930 | 0.8451 | 0.6792 | 0.7527 | 0.7163 | 0.7573 | 0.7674 |
| GBR | 0.7731 | 0.8288 | 0.7136 | 0.7698 | 0.7052 | 0.7581 | 0.7651 |
| RF | 0.7699 | 0.8520 | 0.7055 | 0.6811 | 0.7014 | 0.7420 | 0.7515 |
| LGBM | 0.7059 | 0.8361 | 0.6053 | 0.6420 | 0.6974 | 0.6973 | 0.7121 |
| **Stacking** | 0.8100 | 0.8830 | 0.7092 | 0.8621 | 0.8057 | 0.8140 | **0.8243** |
