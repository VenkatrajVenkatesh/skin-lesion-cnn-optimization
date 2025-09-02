import torch

from utils import load_config, load_and_process_csv

from dataloaders import get_dataloaders

from models import create_model, get_model_info

from training import train_model
 
def run():

    config = load_config()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
    df = load_and_process_csv(config['csv_path'])

    train_loader, val_loader = get_dataloaders(df, config)
 
    model = create_model(config)

    model_info = get_model_info(model)

    print("Model Info:", model_info)
 
    train_model(

        model=model,

        train_loader=train_loader,

        val_loader=val_loader,

        config=config,

        device=device,

        save_path=config.get("save_path", "best_model.pth"),
        
        results_dir = config.get("save_result_path","training_log.json"),

        model_name = config.get("model.arch")

    )
 
if __name__ == "__main__":

    run()

 