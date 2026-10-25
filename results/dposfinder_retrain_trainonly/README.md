# results/dposfinder_retrain_trainonly/

DposFinder, retrained on the thesis training split. Produced by `../../src/dposfinder_retrain.py`. **This is the run reported in the thesis**; `../dposfinder_retrain/` holds a superseded run that also included the validation partition.

## Files

- `train_set.fasta`, `test_set.fasta` — the dataset converted into DposFinder's expected format (1,515 and 176 sequences).
- `test_set_result.tsv` — per-sequence predictions from DposFinder's own predict mode, read by `../../src/benchmark_comparison.py`.

## Notes

- Retraining runs in a separate conda environment (`environment-dposfinder.yml`), since DposFinder's dependencies conflict with the main one.
- Sequences longer than 1024 residues are truncated beforehand (121 in train, 13 in test), matching the token cap applied to the thesis embeddings.
- DposFinder's `final_train` mode has no validation phase: it trains for a fixed 15 epochs, evaluates each epoch on the test set, and keeps the final epoch's weights. Its learning-rate scheduler is driven by that test loss, which is a deviation documented in the thesis methodology.
- `final_train` only prints aggregate metrics, so DposFinder's own predict mode must be run afterwards to obtain per-sequence predictions. `run_pipeline.py` does this automatically.
- Benchmark result: F1 0.9828, PR-AUC 0.9991.