"""

Load the configuration and validate the public EEG input manifest.

Federico Ramírez-Toraño
28/10/2025

"""

# Imports
import json
import pathlib

from sEEGnal.io.recordings import validate_recordings


QUICKSTART_ROOT = pathlib.Path(__file__).resolve().parents[1]
CONFIG_PATH = pathlib.Path(__file__).resolve().with_name('config.json')


def load_config():
    """Load the quickstart configuration and resolve its data paths."""

    with CONFIG_PATH.open('r', encoding='utf-8') as file:
        config = json.load(file)

    data_root = pathlib.Path(config['path']['data_root']).expanduser()
    if not data_root.is_absolute():
        data_root = QUICKSTART_ROOT / data_root
    data_root = data_root.resolve()

    config['path']['data_root'] = str(data_root)
    config['path']['sourcedata'] = str(data_root / 'sourcedata' / 'eeg')
    config['path']['recordings'] = str(data_root / 'recordings.tsv')

    return config


def init(config=None):
    """Load configuration and validate every declared source recording."""

    if config is None:
        config = load_config()

    recordings = validate_recordings(config)
    return config, recordings
