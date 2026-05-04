import yaml
from ultralyrics import YOLO

def ultralytics_train(config_path, data_yaml_path):      
    with open(config_path) as f:                                                   
          config = yaml.safe_load(f)                                                   
   
    model = YOLO(config["weights"])                                                  
    model.train(                                                                   
          data=data_yaml_path,                                                       
          epochs=config.get("epochs", 100),
          imgsz=config.get("imgsz", 640),                                              
          batch=config.get("batch", 16),
          lr0=config.get("lr0", 0.01),                                                 
          optimizer=config.get("optimizer", "SGD"),                                  
          device=config.get("device", 0),                                              
          project=config.get("project", "experiments"),
          name=config.get("name", "run"),                                              
      ) 
