"""Backward-compatible entry point; application code lives in ``src``."""
from src.cli import main, parse_args


def prepare_data(app_config):
    """Retain the historical import used by research scripts."""
    from src.data_pipeline import prepare_data as prepare

    return prepare(app_config)


def load_prepared(app_config):
    """Retain the historical import used by research scripts."""
    from src.data_pipeline import load_prepared as load

    return load(app_config)


if __name__ == "__main__":
    main()
