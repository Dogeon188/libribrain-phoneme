import click
from pathlib import Path
import os

os.environ["PNPL_REMOTE_CONSTANTS_URL"] = "file:////home/dogeon/libribrain_phoneme/constants.json"


@click.command()
@click.argument("mode", type=click.Choice(['train', 'predict', 'submit', 'viz'], case_sensitive=False))
@click.argument("config", type=str)
@click.option("--run-id", type=int, help="Run ID for the hyperparameter optimization.")
def main(mode, config, run_id):
    if mode == 'train':
        from libribrain_experiments.hpo import main as hpo_main
        config_base = Path(f'./configs/phoneme/{config}')
        hpo_main(
            config=config_base / "base-config.yaml",
            search_space=config_base / "search-space.yaml",
            run_index=run_id,
            run_name=config,
            project_name="libribrain-phoneme"
        )
    elif mode == 'predict':
        from libribrain_experiments.predict import main as predict_main
        ckpt_base_name = f"{config}-hpo-{run_id}"
        ckpt_pattern = f"best-*-{ckpt_base_name}-*.ckpt"
        ckpts = list(Path('./out').rglob(ckpt_pattern))
        if not ckpts:
            raise FileNotFoundError(f"No checkpoints found matching pattern: {ckpt_pattern}")
        ckpt_postfix = max(
            [ckpt.stem.split('val_f1_macro=')[-1] for ckpt in ckpts],
            key=lambda x: float(x.split('-')[-1])
        )
        ckpt = next(ckpt for ckpt in ckpts if f"val_f1_macro={ckpt_postfix}" in ckpt.stem)
        print(f"Using checkpoint: {ckpt}")
        config_base = Path(f'./configs/phoneme/{config}')
        config_path = config_base / "submission.yaml"
        predict_main(config_path, ckpt)
    # elif mode == 'submit':
    #     from libribrain_experiments.submit import main as submit_main
    #     submit_main(config)
    elif mode == 'viz':
        from viz import main as viz_main
        result_pattern = f"val-best-{config}-hpo-{run_id}/results.json"
        result_path = next(Path('./out').rglob(result_pattern), None)
        if not result_path:
            raise FileNotFoundError(f"No results found matching pattern: {result_pattern}")
        print(f"Using results file: {result_path}")
        viz_main(result_path)
        
    else:
        raise ValueError(
            "Invalid mode. Choose from 'train', 'predict', or 'submit'.")


if __name__ == "__main__":
    main()
