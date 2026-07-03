# Final artifacts

Best training run:

`runs/detect/waldo_tiled_416_center_neg05_s_80_tuned`

Best checkpoint:

`final_artifacts/best_model/best.pt`

Validation metrics from the best epoch in `results.csv`:

- epoch: 55
- precision: 0.85744
- recall: 0.50774
- mAP50: 0.54352
- mAP50-95: 0.22785

Checkpoint SHA256:

`4048BA65CAFA1417479259E7FF1EED92AFE1A4449FC748EECD83D9A7C3468B4F`

The `scripts` folder contains the core project code needed for tiling, training,
prediction, and evaluation.
