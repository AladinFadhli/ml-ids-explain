# Expliquer un détecteur d'intrusion — CIC-IDS2017

Projet de module *Machine Learning pour la cyber-sécurité*, TEK-UP, ING-5-J-SSIRF-D.
Binôme : Oussama Hamada, Aladine Fadhli. Sujet : cahier des charges 1.

**Question :** le détecteur apprend-il des comportements d'attaque, ou des artefacts de collecte ?

## Run
```bash
pip install -r requirements.txt
# put the CIC-IDS2017 CSVs in data/raw/  (see data/README.md)
jupyter notebook notebooks/projet_ids_explain.ipynb      # Run All, top to bottom
```
Colab: open the notebook, set `REPO_URL` (and `IDS_DATA_DIR` if the data is on Drive) in the first cell, Run All.
Expect roughly 30–60 min on Colab's free CPU (tuning + SHAP on the random forest).
Smaller/faster: lower `MAX_ROWS` and `TUNE_MAX_ROWS` in `src/config.py`.

## Layout
| Path | Content |
|---|---|
| `notebooks/projet_ids_explain.ipynb` | the deliverable: code + report |
| `src/config.py` | seed, sizes, identifier and suspect columns |
| `src/data.py` | load, clean (before split), sample, split 60/20/20, train-only feature selection, leak check |
| `src/models.py` | LogReg baseline, Random Forest, XGBoost; 5-fold CV on train, PR-AUC |
| `src/evaluate.py` | threshold on validation, metrics, PR curves, alert load, recall per family |
| `src/explain.py` | 3 global importances, 5 SHAP alerts, ablation, stability |
| `tests/make_fake_data.py` | synthetic CSVs shaped like CIC-IDS2017, to smoke-test the pipeline |

## Leak-free rules
- Cleaning before the split uses no statistic of the data (IDs, inf/NaN, duplicates).
- Constant/duplicate columns, scaler, tuning, threshold: learned on train / validation only.
- Test is scored once with frozen models; `leakage_check` confirms no test row also appears in train.

## Smoke test (no real data)
```bash
python tests/make_fake_data.py /tmp/fake 30000
IDS_DATA_DIR=/tmp/fake jupyter nbconvert --to notebook --execute notebooks/projet_ids_explain.ipynb --output /tmp/out.ipynb
```

## References
- Engelen, Rimmer, Joosen, *Troubleshooting an Intrusion Detection Dataset: the CICIDS2017 Case Study*, IEEE SPW 2021.
- Sharafaldin, Lashkari, Ghorbani, *Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization*, ICISSP 2018.
