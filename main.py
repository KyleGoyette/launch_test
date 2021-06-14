import wandb
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=0)
parser.add_argument("--epochs", type=float, default=5)
args = parser.parse_args()

wandb.init(project="launch-demo")
wandb.run.config["val1"] = 10
wandb.run.config["val2"] = 20
wandb.run.config["lr"] = args.lr
wandb.run.config["epochs"] = args.epochs
wandb.log({"lr": args.lr})
wandb.log({"eopchs": args.epochs})
