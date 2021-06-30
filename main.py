import wandb
import argparse
import numpy

parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=0.1)
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--seed", type=int, default=10)
args = parser.parse_args()

wandb.init(project="launch-demo")
wandb.run.config["nested"] = {"super_nested": "jerk"}
wandb.run.config["lr"] = args.lr
wandb.run.config["epochs"] = args.epochs
wandb.run.config["accs"] = args.accs
for i in range(wandb.run.config.epochs):
    wandb.log({"loss": 1000000/((i+1)*wandb.config.lr)})
    wandb.log({"acc": 1 - 2/(i+1)})

    
