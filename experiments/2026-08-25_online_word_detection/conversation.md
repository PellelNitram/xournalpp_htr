# Online Word Stroke Detection — Experiment Conversation Summary

**Date**: 2026-08-25 to 2026-08-26
**Branch**: `improve_word_detector_net`
**Working directory**: `experiment_online_detection/`

---

## 1. Goal

Explore **online word stroke detection** — grouping raw strokes (x, y coordinate arrays) into words — directly from stroke coordinates, without rendering to images. This contrasts with the current offline pipeline (render strokes to image, run CNN word detector, map bounding boxes back to strokes).

## 2. Setup and Constraints

- **Dataset**: `PellelNitram/xournalpp_htr_benchmark` on HuggingFace. Only **6 pages** with ground truth `stroke_indices` per word annotation.
- **Evaluation protocol**: Leave-one-page-out cross-validation (LOPO, 6 folds). Tune hyperparameters on 5 training pages, evaluate on the held-out page.
- **Self-contained**: Everything lives in `experiment_online_detection/` with its own `uv` environment. No path dependencies on the parent `xournalpp_htr` project. Needed code (benchmark loading, document parsing) is copied directly into the notebooks.
- **Notebook as single source**: After initially creating separate `.py` files (`data.py`, `features.py`, `evaluate.py`, `baseline_spatial.py`, `run.py`), the user chose to consolidate everything into Jupyter notebooks and delete the scripts.

## 3. Task Definition

- **Input**: All strokes on a page (list of `Stroke` objects with x, y coordinate arrays).
- **Output**: For each stroke, which word it belongs to (cluster assignment).
- **Evaluation metrics**: Jaccard-based cluster matching (Precision/Recall/F1 at word level, stroke accuracy). A predicted cluster matches a GT word if Jaccard index of stroke sets >= 0.5.
- **Ground truth**: Parsed from `.gt.json` files, filtering to `class in {"word", "digit", "mathematical_expression"}`.

## 4. Experiments Conducted

### 4.1 Spatial Proximity Baseline (`experiment.ipynb`)

- **Approach**: Compute bounding box per stroke, compute pairwise min-edge distances between bboxes, cluster with agglomerative clustering (single linkage), tune distance threshold on training pages.
- **Sections**: Data Loading, Visualize Dataset, Feature Extraction, Evaluation Metrics, Spatial Proximity Baseline, LOPO CV, Results Table, Threshold Sensitivity.
- **Result**: F1 = 0.733 +/- 0.178

### 4.2 Learned Pairwise Classifier (`experiment_pairwise_classifier.ipynb`)

- **Approach**: Compute 13 pairwise features for all stroke pairs, train classifier (GradientBoosting, RandomForest, LogisticRegression), predict same-word / different-word, form clusters via connected components.
- **13 pairwise features**: `min_point_dist`, `bbox_edge_dist`, `centroid_dist`, `horizontal_gap`, `vertical_gap`, `horizontal_overlap_ratio`, `vertical_overlap_ratio`, `centroid_angle`, `stroke_index_diff`, `area_ratio`, `aspect_ratio_diff`, `path_length_ratio`, `n_points_ratio`.
- **Class imbalance**: Up to 110:1 ratio of different-word to same-word pairs on dense pages.
- **Sections**: 10 sections including feature importance and visualization.
- **Result**: GB best at F1 = 0.719 +/- 0.288. Connected components fragility: one wrong edge merges entire words together.

### 4.3 Affinity Clustering (`experiment_affinity_clustering.ipynb`)

- **Approach**: Use GB predicted probabilities as an affinity matrix, run agglomerative clustering with average linkage and a distance threshold (using `1 - probability` as distance).
- **Result**: F1 = 0.715 +/- 0.254

### 4.4 Render + Detect (`experiment_render_detect.ipynb`)

- **Approach**: Render strokes to binary image (PIL), find connected components (OpenCV), map components back to strokes based on which component each stroke's pixels land in. No neural network — pure image processing.
- **Dependencies added**: `opencv-python`, `Pillow`, `scipy`.
- **Result**: F1 = 0.749 +/- 0.149

### 4.5 Hybrid Classifier-Filtered Spatial Clustering (`experiment_hybrid_filtering.ipynb`) — BEST

- **Approach**: Two-stage pipeline:
  1. Train GB classifier on pairwise features, predict P(same word) for all pairs.
  2. Build filtered distance matrix: use bbox distance where P >= `prob_threshold`, else set to 1e6 (infinity).
  3. Run AgglomerativeClustering with `metric="precomputed"`, `linkage="single"`, `distance_threshold=dist_threshold`.
- **Two hyperparameters** tuned jointly via grid search per CV fold:
  - `prob_threshold`: min P(same word) to keep a pair (candidates: `np.arange(0.1, 0.9, 0.1)`)
  - `dist_threshold`: clustering distance cutoff (candidates: `np.arange(1.0, 50.0, 2.0)`)
- **Key function**: `cluster_with_filtered_matrix()` — builds bbox distance matrix, masks pairs below probability threshold, runs agglomerative clustering.
- **Why it works**: The classifier filters out clearly wrong pairs (solving the connected-components fragility), then spatial clustering handles the final grouping (robust with limited training data). This solves the tension: ML alone struggles with only 6 pages, spatial alone can't distinguish dense adjacent words.
- **Result**: F1 = 0.894 +/- 0.102, Precision = 0.875 +/- 0.153, Recall = 0.927 +/- 0.069, Stroke Acc = 0.920 +/- 0.105

## 5. Results Summary

| Method                    | Precision         | Recall            | F1                | Stroke Acc        |
|---------------------------|-------------------|-------------------|-------------------|-------------------|
| Spatial Baseline          | 0.762 +/- 0.175   | 0.734 +/- 0.213   | 0.733 +/- 0.178   | 0.641 +/- 0.259   |
| GB + Conn. Components     | 0.738 +/- 0.219   | 0.741 +/- 0.318   | 0.719 +/- 0.288   | 0.768 +/- 0.286   |
| Affinity Clustering       | ~0.715             | ~0.715             | 0.715 +/- 0.254   | —                 |
| Render + Detect           | ~0.749             | ~0.749             | 0.749 +/- 0.149   | —                 |
| **Hybrid (GB + Spatial)** | **0.875 +/- 0.153**| **0.927 +/- 0.069**| **0.894 +/- 0.102**| **0.920 +/- 0.105**|

## 6. ONNX Export

Added as Section 9 in `experiment_hybrid_filtering.ipynb`:

- **What ships as ONNX**: `StandardScaler` + `GradientBoostingClassifier` combined in a sklearn `Pipeline`, exported via `skl2onnx.to_onnx()`.
- **What stays as Python**: Feature extraction (13 pairwise features) and clustering logic (`cluster_with_filtered_matrix`) — these are pure numpy/sklearn and don't benefit from ONNX.
- **Training**: Trained on all 6 pages (no hold-out — the CV already validated the approach).
- **Export file**: `hybrid_word_detector.onnx` (213 KB).
- **Verification**: The export cell compares sklearn vs ONNX predictions on 100 samples and asserts max probability difference < 1e-5.
- **ONNX model I/O**: Input `X` is `(N, 13)` float32 pairwise features. Output is class probabilities; column 1 is P(same word).
- **Dependencies added**: `skl2onnx>=1.16.0`, `onnxruntime>=1.17.0` in `pyproject.toml`.

## 7. File Inventory

```
experiment_online_detection/
├── pyproject.toml                              # uv project config (self-contained)
├── uv.lock                                     # locked dependencies
├── experiment.ipynb                            # Exp 1: Spatial proximity baseline
├── experiment_pairwise_classifier.ipynb        # Exp 2: Learned pairwise classifier (GB/RF/LR)
├── experiment_affinity_clustering.ipynb        # Exp 3: Affinity clustering
├── experiment_render_detect.ipynb              # Exp 4: Render + connected components
├── experiment_hybrid_filtering.ipynb           # Exp 5: Hybrid GB filter + spatial (BEST) + ONNX export
├── hybrid_word_detector.onnx                   # Exported ONNX model (213 KB)
├── results_html/                               # All 5 notebooks executed as HTML
│   ├── experiment.html
│   ├── experiment_pairwise_classifier.html
│   ├── experiment_affinity_clustering.html
│   ├── experiment_render_detect.html
│   └── experiment_hybrid_filtering.html
├── .venv/                                      # uv virtual environment
└── __pycache__/
```

## 8. Dependencies (`pyproject.toml`)

```
numpy>=1.24.0, scikit-learn>=1.3.0, matplotlib>=3.7.0, tabulate>=0.9.0,
huggingface_hub>=0.20.0, beautifulsoup4>=4.12.0, lxml>=4.9.0,
jupyter>=1.0.0, ipykernel>=6.0.0, opencv-python>=4.8.0, Pillow>=10.0.0,
scipy>=1.10.0, skl2onnx>=1.16.0, onnxruntime>=1.17.0
```

## 9. Key Technical Details

### Data loading (copied into each notebook)

- Download via `huggingface_hub.snapshot_download("PellelNitram/xournalpp_htr_benchmark")`.
- Parse `.xopp`/`.xoj` gzipped XML with BeautifulSoup + lxml (with `XMLParsedAsHTMLWarning` suppressed).
- Parse `.gt.json` for annotations, filtering to `_TEXT_CLASSES = {"word", "digit", "mathematical_expression"}`.
- Compute global stroke indices by offsetting within layers on a page.
- Produce `PageData(page_id, strokes, word_groups)` per page.

### Evaluation (`evaluate_grouping()`)

- Greedy best-match between predicted clusters and GT words using Jaccard index.
- A match requires Jaccard >= 0.5.
- Precision = matched / predicted clusters, Recall = matched / GT words.
- Stroke accuracy = fraction of word strokes in a Jaccard-matched cluster.

### Hybrid model inference pipeline (for shipping)

1. Extract 13 pairwise features for all stroke pairs on a page.
2. Run features through ONNX model to get P(same word) per pair.
3. Build filtered distance matrix: bbox distance where P >= `prob_threshold`, else 1e6.
4. Run `AgglomerativeClustering(metric="precomputed", linkage="single", distance_threshold=dist_threshold)`.
5. Output: list of stroke index sets, one per predicted word.

Note: `prob_threshold` and `dist_threshold` are fixed at inference time — they were tuned during CV but need to be hardcoded when deploying. Typical best values from CV folds should be used (check the per-fold results in the notebook output).

## 10. Bugs and Fixes Encountered

- **SSL certificate error**: `[SSL: CERTIFICATE_VERIFY_FAILED]` when downloading HuggingFace dataset. Resolved on retry.
- **matplotlib `get_cmap` removed**: `AttributeError: module 'matplotlib.cm' has no attribute 'get_cmap'` in matplotlib 3.11+. Fixed by changing `plt.cm.get_cmap("tab20")` to `plt.colormaps["tab20"]`.
- **XML parser warning**: `XMLParsedAsHTMLWarning` from BeautifulSoup. Suppressed with `warnings.filterwarnings("ignore", ...)`.

## 11. Possible Next Steps

- **Deploy the hybrid model**: Integrate into the main `xournalpp_htr` pipeline as an alternative to the offline WordDetectorNet path. Would need to hardcode optimal `prob_threshold` and `dist_threshold` and upload `hybrid_word_detector.onnx` to HuggingFace.
- **More training data**: The 6-page benchmark is very small. More annotated pages would likely improve the classifier and reduce per-fold variance.
- **Feature engineering**: Additional pairwise features (e.g., temporal/stroke-order features, ink density between strokes) might help.
- **Threshold selection strategy**: Instead of hardcoding thresholds, could calibrate on a validation set or use an adaptive scheme.
- **Comparison with WordDetectorNet**: Run the current offline pipeline on the same 6 pages for a direct head-to-head.
