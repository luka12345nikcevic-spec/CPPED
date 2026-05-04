import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))


DEFAULT_CONFIG = Path(__file__).resolve().parent / "config" / "experiments.yaml"
CONFIG_DIR = Path(__file__).resolve().parent / "config"
PUBLIC_DATASETS_DIR = REPO_ROOT / "data" / "public_datasets"


def infer_dataset_name(data_yaml_path):
    data_yaml_path = Path(data_yaml_path).resolve()
    parts = data_yaml_path.parts

    if "public_datasets" in parts:
        index = parts.index("public_datasets")
        if index + 1 < len(parts):
            return parts[index + 1]

    return data_yaml_path.parent.name


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLOv8 using the shared Ultralytics trainer.")
    parser.add_argument("--data-yaml", help="Path to the dataset data.yaml file.")
    parser.add_argument("--dataset-name", help="Dataset name used for the output weights path.")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Path to the YOLOv8 training config.")
    return parser.parse_args()


def discover_public_datasets():
    datasets = []

    for dataset_dir in sorted(PUBLIC_DATASETS_DIR.iterdir()):
        if not dataset_dir.is_dir():
            continue

        data_yamls = sorted(dataset_dir.rglob("data.yaml"))
        if data_yamls:
            datasets.append((dataset_dir.name, data_yamls[0]))

    return datasets


def discover_configs():
    return sorted(CONFIG_DIR.rglob("*.yaml"))


def train_all_public_datasets():
    datasets = discover_public_datasets()
    configs = discover_configs()
    from src.models.ultralytics_train import ultralytics_train

    if not datasets:
        raise FileNotFoundError(f"No data.yaml files found under {PUBLIC_DATASETS_DIR}")

    if not configs:
        raise FileNotFoundError(f"No config YAML files found under {CONFIG_DIR}")

    for config_path in configs:
        for dataset_name, data_yaml_path in datasets:
            print(f"Training YOLOv8 config={config_path} dataset={dataset_name}")
            ultralytics_train(config_path, data_yaml_path, dataset_name)


def main():
    if len(sys.argv) == 1:
        train_all_public_datasets()
        return

    args = parse_args()
    if not args.data_yaml:
        raise ValueError("--data-yaml is required when passing command-line arguments")

    from src.models.ultralytics_train import ultralytics_train

    dataset_name = args.dataset_name or infer_dataset_name(args.data_yaml)
    ultralytics_train(args.config, args.data_yaml, dataset_name)


if __name__ == "__main__":
    main()
