from pathlib import Path
from tempfile import TemporaryDirectory
import shutil

import yaml
from ultralytics import YOLO

def _model_dir_from_config(config_path):
    model_dir = Path(config_path).resolve().parent
    while model_dir.name != "config" and model_dir.parent != model_dir:
        model_dir = model_dir.parent

    if model_dir.name == "config":
        return model_dir.parent

    raise ValueError(f"Could not find model directory from config path: {config_path}")

def ultralytics_train(config_path, data_yaml_path, dataset_name):      
    config_path = Path(config_path)

    with open(config_path) as f:                                                   
          config = yaml.safe_load(f) or {}

    model_dir = _model_dir_from_config(config_path)
    run_name = config.get("name", config_path.stem)
    weights_dir = model_dir / "weights" / dataset_name / run_name
    weights_dir.mkdir(parents=True, exist_ok=True)
   
    model = YOLO(config["weights"])                                                  

    with TemporaryDirectory(prefix="cpped_yolo_train_") as temp_dir:
        model.train(
              data=data_yaml_path,                                                       
              epochs=config.get("epochs", 100),
              imgsz=config.get("imgsz", 640),                                              
              batch=config.get("batch", 16),
              lr0=config.get("lr0", 0.01),
              lrf=config.get("lrf", 0.01),
              momentum=config.get("momentum", 0.937),
              weight_decay=config.get("weight_decay", 0.0005),
              warmup_epochs=config.get("warmup_epochs", 3.0),
              optimizer=config.get("optimizer", "SGD"),                                  
              patience=config.get("patience", 50),
              device=config.get("device", 0),
              workers=config.get("workers", 8),
              cos_lr=config.get("cos_lr", False),
              augment=config.get("augment", True),
              project=temp_dir,
              name="run",
              exist_ok=True,
          )

        run_dir = Path(getattr(model.trainer, "save_dir", Path(temp_dir) / "run"))
        trained_weights_dir = run_dir / "weights"
        trained_weights = list(trained_weights_dir.glob("*.pt"))

        if not trained_weights:
            raise FileNotFoundError(f"No .pt files were found in {trained_weights_dir}")

        for weight_file in trained_weights:
            shutil.copy2(weight_file, weights_dir / weight_file.name)
