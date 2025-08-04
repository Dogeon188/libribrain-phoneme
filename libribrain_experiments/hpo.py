from argparse import ArgumentParser
import itertools
from .utils import run_training, get_datasets_from_config, adapt_config_to_data, run_validation, log_results, get_label_counts
import yaml
import wandb
import pytorch_lightning as lightning
import numpy as np
import torch
import time
from lightning.pytorch.accelerators import find_usable_cuda_devices
import click


def update_config_for_single_run(config: dict, run_config: list[tuple[tuple, list]]):
    """
        config: dict
        run_config: list of updates to the config. Each entry maps a keylist to a value e.g. (("optimizer", "config", "lr"), 0.01)
    """
    for key_list, value in run_config:
        current = config
        try:
            for key in key_list[:-1]:
                current = current[key]
            current[key_list[-1]] = value
        except KeyError:
            raise KeyError(
                f"Key list {key_list} not found in config. Config: {config}")
    return config


def runs_configs_from_search_space(search_space: dict[tuple, list]):
    """
        search_space: dict that maps key_list to the list of values to try for that key
        returns: list where each element describes all the hyperparameter updates for a single run
    """
    if (len(search_space) == 0):
        return []
    keys, values = zip(*search_space.items())
    result = []
    for v in itertools.product(*values):
        config = list(zip(keys, v))
        result.append(config)
    return result


def get_run(config, search_space, i):
    run_config = search_space[i]
    return update_config_for_single_run(config, run_config)


def load_search_space(path: str):
    try:
        with open(path, 'r') as f:
            search_space = yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(
            "Search space file not found. Please provide a valid path")
    search_space = parse_search_space(search_space)
    return search_space


def parse_search_space(search_space: dict):
    result = {}
    for key, value in search_space.items():
        new_key = eval(key)
        result[new_key] = value
    return result


def main(config, run_index, search_space, run_name, project_name):

    start_time = time.time()
    try:
        with open(config, 'r') as f:
            config_data = yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(
            "Config file not found. Please provide a valid path")

    search_space_data = load_search_space(search_space)
    run_configs = runs_configs_from_search_space(search_space_data)
    if (run_index is None):
        run_index = np.random.randint(0, len(run_configs))
    config_data = get_run(config_data, run_configs, run_index)

    print("Running config: ", run_index)
    print("Config: ", run_configs[run_index])
    if 'use_tf32' in config_data['general']:
        use_tf32 = config_data['general']['use_tf32']
    else:
        use_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = use_tf32

    if (run_name is None):
        run_name = "hpo-run-" + str(run_index)
    else:
        run_name = run_name + "-hpo-" + str(run_index)
    config_data["general"]["run_name"] = run_name
    if (config_data["general"]["wandb"]):
        if (project_name is None):
            raise ValueError(
                "Please provide a project name for wandb logging")
        wandb.init(project=project_name, name=run_name)
        wandb.define_metric("val_loss", summary="min")
        wandb.define_metric("val_f1_macro", summary="max")
        wandb.define_metric("val_bal_acc", summary="max")
    print("LOADED CONFIGS in ", time.time() - start_time, " seconds")

    start_time = time.time()
    lightning.seed_everything(config_data["general"]["seed"])
    print("SEEDED EVERYTHING in ", time.time() - start_time, " seconds")

    start_time = time.time()
    train_dataset, val_dataset, test_dataset, labels = get_datasets_from_config(
        config_data["data"])
    print("LOADED DATASETS in ", time.time() - start_time, " seconds")

    if ("train_fraction" in config_data["data"]["general"]):
        train_fraction = config_data["data"]["general"]["train_fraction"]
        train_size = int(len(train_dataset) * train_fraction)
        remaining_size = len(train_dataset) - train_size
        train_dataset, _ = torch.utils.data.random_split(
            train_dataset, [train_size, remaining_size])
    print("TRAIN SIZE: ", len(train_dataset))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, shuffle=True, **config_data["data"]["dataloader"])
    val_loader = torch.utils.data.DataLoader(
        val_dataset, **config_data["data"]["dataloader"])
    adapt_config_to_data(config_data, train_loader, labels)

    start_time = time.time()
    print("ADAPTED CONFIG TO DATA in ", time.time() - start_time, " seconds")

    if "best_model_metrics" in config_data["general"]:
        best_model_metric = config_data["general"]["best_model_metrics"]
    else:
        best_model_metric = "val_bal_acc"

    if best_model_metric == "val_loss":
        best_model_metric_mode = "min"
    else:
        best_model_metric_mode = "max"

    start_time = time.time()


    # replace predefined block names with actual blocks
    model_blocks = config_data.get("blocks", None)
    if model_blocks is not None:
        for i, module in enumerate(config_data["model"]):
            if isinstance(module, dict) and list(module.keys())[0] in model_blocks:
                block_name = list(module.keys())[0]
                config_data["model"] = config_data["model"][:i] + model_blocks[block_name] + config_data["model"][i+1:]
        
    trainer, best_module, module = run_training(
        train_loader, val_loader, config_data, len(labels), best_model_metric=best_model_metric, best_model_metric_mode=best_model_metric_mode)
    print("TRAINED MODEL in ", time.time() - start_time, " seconds")

    samples_per_class = get_label_counts(train_loader, len(labels))

    """result, y, preds, logits = run_validation(
        val_loader, module, labels, avg_evals=[5, 100], samples_per_class=samples_per_class)
    start_time = time.time()
    print("VALIDATED LAST MODEL in ", time.time() - start_time, " seconds")

    log_results(result, y, preds, logits,
                config_data["general"]["output_path"], "last-" + str(run_name), hpo_config=run_configs[run_index], trainer=trainer)
    start_time = time.time()
    print("LOGGED LAST RESULTS in ", time.time() - start_time, " seconds")"""
    del module

    best_module = best_module.to(find_usable_cuda_devices()[0])
    start_time = time.time()
    result, y, preds, logits = run_validation(
        val_loader, best_module, labels, samples_per_class=samples_per_class)
    print("VALIDATED MODEL in ", time.time() - start_time, " seconds")

    start_time = time.time()
    log_results(result, y, preds, logits,
                config_data["general"]["output_path"], "val-best-" + str(run_name))
    print("LOGGED BEST RESULTS in ", time.time() - start_time, " seconds")

    if test_dataset is not None:
        print("Validating on test set")
        test_loader = torch.utils.data.DataLoader(
            test_dataset, **config_data["data"]["dataloader"])
        start_time = time.time()
        result, y, preds, logits = run_validation(
            test_loader, best_module, labels, samples_per_class=samples_per_class, name="test")
        print("VALIDATED MODEL in ", time.time() - start_time, " seconds")

        start_time = time.time()
        log_results(result, y, preds, logits,
                    config_data["general"]["output_path"], "test-best-" + str(run_name))
        print("LOGGED BEST RESULTS in ", time.time() - start_time, " seconds")


@click.command()
@click.option("--config", type=click.Path(exists=True), required=True, help="Path to the config YAML file.")
@click.option("--run-index", type=int, default=None, help="Index of the run to execute. If none, random run will be chosen.")
@click.option("--search-space", type=click.Path(exists=True), required=True, help="Path to the search space YAML file.")
@click.option("--run-name", type=str, default=None, help="Name for this run.")
@click.option("--project-name", type=str, default="libribrain-experiments", help="wandb project name.")
def cli(config, run_index, search_space, run_name, project_name):
    """
    Command line interface for running hyperparameter optimization.
    """
    main(config, run_index, search_space, run_name, project_name)


if __name__ == "__main__":
    cli()
