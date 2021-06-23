import wandb
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=0.1)
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--accs", action="store_true")
args = parser.parse_args()

wandb.init(project="launch-demo")
wandb.run.config["lr"] = args.lr
wandb.run.config["epochs"] = args.epochs
wandb.run.config["accs"] = args.accs
for i in range(wandb.run.config.epochs):
    wandb.log({"loss": 1/((i+1)*wandb.config.lr)})
    if wandb.run.config.accs:
        wandb.log({"acc": 1 - 1/(i+1)})

    
