# LibriBrain 2025

The code is modified from [neural-processing-lab/libribrain-experiments](https://github.com/neural-processing-lab/libribrain-experiments). For details of the competition, see the [LibriBrain 2025 website](https://neural-processing-lab.github.io/2025-libribrain-competition/).


## Installation

Before starting the project, please run:

```sh
uv sync
ln -s /home/share/data/libribrain2025 $(pwd)/data
ln -s /home/share/data/libribrain_phoneme_preprocessed $(pwd)/data_preprocessed
```

If you don't have `uv` installed, follow the instructions at [astral-sh/uv](https://github.com/astral-sh/uv).

## Experiment Configuration

Configuration files can be found in `configs/<task>/<config-name>/`. You can change the model structure as well as the data parameters in these files.

**Important Configuration Notes:**

Before running the project, make sure to update the configuration files with the correct local paths:

- **`data_path`**: Specify the paths for your training, validation, and testing datasets.
- **`output_path`**: Set this to the directory where output results (e.g., logs, predictions) will be saved.
- **`checkpoint_path`**: Define the location where model checkpoints should be stored.

## Running Experiments

The project entrypoint is [`main.py`](./main.py):

```bash
uv run main.py <mode> <config-name> <run-id>
```

Available modes:

- `train`: run HPO/training for `configs/phoneme/<config-name>/`
- `predict`: load the best checkpoint for that run and generate holdout predictions
- `val`: load the best checkpoint for that run and evaluate on validation data
- `viz`: visualize the validation results JSON for that run

`<config-name>` is the directory name under `configs/phoneme/`, for example `baseline-norm`.
`<run-id>` is the run index used to distinguish repeated runs of the same config.

## Training a Model

Example:

```bash
uv run main.py train baseline-norm 0
```

This uses:

- `configs/phoneme/baseline-norm/base-config.yaml`
- `configs/phoneme/baseline-norm/search-space.yaml`

## Generating Predictions

To generate holdout predictions from the best checkpoint for a run:

```bash
uv run main.py predict baseline-norm 0
```

This automatically finds the best checkpoint matching `baseline-norm-hpo-0` under `./out/` and uses `configs/phoneme/baseline-norm/submission.yaml` for prediction settings. The predictions are saved as a CSV by the prediction pipeline.

## Validating a Trained Model

To run validation for the best checkpoint of a run:

```bash
uv run main.py val baseline-norm 0
```

## Visualizing Validation Results

To visualize the saved validation results for a run:

```bash
uv run main.py viz baseline-norm 0
```

## Submitting Predictions

To submit your predictions, use the following command:

```bash
sh submit.sh <pred-csv-file>
```

`submit.sh` is only for uploading an already generated predictions CSV to EvalAI. Training, validation, prediction generation, and visualization all go through `main.py`.

## Layer Saliency Clustermap

Use `generate_layer_saliency_clustermap.py` to estimate which model sublayers are most sensitive to each phoneme class for a saved checkpoint.

Example:

```bash
uv run python generate_layer_saliency_clustermap.py \
  --config configs/phoneme/stft/base-config.yaml \
  --checkpoint out/phoneme-stft/best-val_bal_acc-stft-hpo-9-epoch=09-val_f1_macro=0.4111.ckpt \
  --output-dir saliency_maps/stft-0.4111-layer-clustermap \
  --batch-size 32 \
  --num-workers 4
```

The script writes one set of files for each split (`validation` and `test`):

- `*_layer_saliency_raw.csv`: mean absolute gradient score for each `layer.sublayer` and phoneme
- `*_layer_saliency_rowminmax.csv`: row-wise normalized version of the raw table
- `*_layer_saliency_clustermap.png`: clustered heatmap rendered from the normalized table
- `*_layer_saliency_summary.json`: metadata such as checkpoint path, row names, phoneme order, and label counts

How to read the figure:

- Each row is a model component discovered from the module tree, named in `<layer>.<sublayer>` form when possible, for example `res0.conv2d0` or `classifier.linear1`.
- Each column is a phoneme class.
- Each cell shows the relative saliency of that sublayer for that phoneme. Higher values mean the target logit is more sensitive to activations at that sublayer.
- The PNG uses the row-wise normalized scores, so colors are best used to compare phonemes within the same row, not absolute magnitudes across different rows.
- The dendrograms reorder rows and columns by similarity. Nearby rows have similar phoneme-saliency patterns, and nearby columns are phonemes that trigger similar sublayer sensitivity profiles.
