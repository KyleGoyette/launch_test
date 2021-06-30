import wandb
import argparse
import math
import random

parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=0.1)
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--seed", type=int, default=10)
args = parser.parse_args()

run = wandb.init(
        project="launch-demo",
        config={
            "learning_rate": args.lr,
            "batch_size": 128,
            "momentum": 0.1,
            "dropout": 0.2,
            "architecture": "CNN",
            "epochs": args.epochs,
            "seed": args.seed
        })

random.seed(run.config.seed)
displacement1 = random.random()
displacement2 = random.random()
for step in range(run.config.epochs):
    wandb.log({
        "acc": 0.04 * (math.log(1 + step + random.random()) + random.random() * run.config.learning_rate + random.random() + displacement1 + random.random() * run.config.momentum),
        "val_acc": 0.04 * (math.log(1 + step + random.random()) + random.random() * run.config.learning_rate - random.random() + displacement1),
        "loss": 0.04 * (4 - math.log(1 + step + random.random()) + random.random() * run.config.momentum + random.random() + displacement2),
        "val_loss": 0.04 * (5 - math.log(1 + step + random.random()) + random.random() * run.config.learning_rate - random.random() + displacement2),
    })