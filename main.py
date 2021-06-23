import wandb
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=0)
parser.add_argument("--epochs", type=float, default=5)
parser.add_argument("--flag", action="store_true")
args = parser.parse_args()

wandb.init(project="launch-demo10")
wandb.run.config["val1"] = 10
wandb.run.config["val2"] = 20
wandb.run.config["lr"] = args.lr
wandb.run.config["flag"] = args.flag
wandb.run.config["epochs"] = args.epochs
for i in range(100):
    #time.sleep(0.5)
    wandb.log({"lr": wandb.config.lr})
    wandb.log({"eopchs": args.epochs})
