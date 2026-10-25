# results/phagedpo_retrain/

PhageDPO, retrained on the thesis dataset. Produced by `../../src/phagedpo_retrain.py`. PhageDPO is the tool this thesis directly builds on.

## Files

- `phagedpo_features.pkl` — cached hand-crafted descriptors (length, amino acid counts, physicochemical properties, CTD, dipeptide composition).
- `best_params.json`, `cv_results.csv` — the configuration selected by PhageDPO's own grid search (10-fold CV, optimising MCC) and the full search results.
- `phagedpo_retrained_svm.pkl` — the refitted support vector classifier.
- `phagedpo_retrained_predictions.csv`, `test_metrics.json` — test-set predictions and metrics.

## Notes

- Standardisation and ANOVA-based feature selection are steps inside the pipeline, so they are refitted on each fold's training portion alone; the test set takes no part in model selection.
- Class probabilities come from the classifier's internal calibration, fitted by a further cross-validation on the training data.
- Sequences containing non-standard residues are dropped before featurisation (43 of the 1,515 training sequences; none in the test set).
- Benchmark result: F1 0.9107, PR-AUC 0.9675.