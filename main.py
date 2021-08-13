import wandb
import argparse
import math
import numpy as np
import random


parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=0.1)
parser.add_argument("--epochs", type=int, default=15)
parser.add_argument("--seed", type=int, default=10)
args = parser.parse_args()

run = wandb.init(
        project="launch-arti-demo",
        job_type="train",
        config={
            "learning_rate": args.lr,
            "batch_size": 128,
            "momentum": 0.1,
            "dropout": 0.2,
            "architecture": "CNN",
            "epochs": args.epochs,
            "seed": args.seed
        })

np.random.seed(run.config.seed)

dataset = run.use_artifact('launch-arti-demo/my-orig-dataset:latest')
run.config.dataset = dataset

if run.config.dataset.name == 'my-new-dataset:v0':
    const = 0.5
else:
    const = 0

displacement1 = np.random.uniform(0, 0.5) + const
displacement2 = np.random.uniform(0.5, 1) - const

for step in range(5):
    wandb.log({
        "acc": 0.04 * (math.log(1 + step + random.random()) + random.random() * run.config.learning_rate + random.random() + displacement1 + random.random() * run.config.momentum),
        "val_acc": 0.04 * (math.log(1 + step + random.random()) + random.random() * run.config.learning_rate - random.random() + displacement1),
        "loss": 0.04 * (4 - math.log(1 + step + random.random()) + random.random() * run.config.momentum + random.random() + displacement2),
        "val_loss": 0.04 * (5 - math.log(1 + step + random.random()) + random.random() * run.config.learning_rate - random.random() + displacement2),
        "train_step": step*25
    })
