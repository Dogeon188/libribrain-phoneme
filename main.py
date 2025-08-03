import click
from pathlib import Path
import os

os.environ["PNPL_REMOTE_CONSTANTS_URL"] = "file:////home/dogeon/libribrain_phoneme/constants.json"


@click.command()
@click.argument("mode", type=click.Choice(['train', 'predict', 'submit'], case_sensitive=False))
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
        predict_main(config)
    # elif mode == 'submit':
    #     from libribrain_experiments.submit import main as submit_main
    #     submit_main(config)
    else:
        raise ValueError(
            "Invalid mode. Choose from 'train', 'predict', or 'submit'.")


if __name__ == "__main__":
    main()
