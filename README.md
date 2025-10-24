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

## Training a Model

Use the following command format to execute an experiment:

```bash
TASK=<task>  # "speech" or "phoneme"
CONFIG_NAME=<config-name> # the name of the configuration at configs/<task>/<config-name>/
RUN_NAME=<run-name> # custom name for the run, will be shown in W&B dashboard, by default = <config-name>
RUN_ID=<run-id> # run index, used to distinguish different runs with the same name, by default = 0

sh run.sh
```

## Generating Predictions

To generate predictions for the holdout dataset, use the following command:

```bash
TASK=<task>  # "speech" or "phoneme"
CONFIG_NAME=<config-name> # the name of the configuration at configs/<task>/<config-name>/
RUN_ID=<run-id> # run index, used to distinguish different runs with the same name, by default = 0

sh submission.sh
```

This will automatically load the best checkpoint (according to validation F1 score) and generate predictions for the holdout dataset. The predictions will be saved in a CSV file named `preds_<ckpt-path>.csv`.

## Submitting Predictions

To submit your predictions, use the following command:

```bash
sh submit.sh <pred-csv-file>
```
