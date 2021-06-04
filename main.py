import wandb
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=0)
parser.add_argument("--epochs", type=float, default=5)
args = parser.parse_args()

wandb.init(project="launchtest1")
wandb.log({"lr": args.lr})
wandb.log({"eopchs": args.epochs})
