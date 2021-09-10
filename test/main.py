import wandb
import argparse
import math
import numpy as np
import random
import os
import json

parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=0.1)
parser.add_argument("--epochs", type=int, default=15)
parser.add_argument("--seed", type=int, default=10)
args = parser.parse_args()


# os.environ["WANDB_LAUNCH_CONFIG_PATH"] = fpath
# os.environ["WANDB_LAUNCH"] = "True"


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

#dataset = run.use_artifact('my-orig-dataset:latest')
#model = run.use_artifact('my-orig-dataset:latest')
#print("FIUC", run.config._items)
# if dataset.name == 'my-new-dataset:v0':
#     print("Using my-new-dataset")
#     const = 0.5
# else:
    # const = 0
const = 1
displacement1 = np.random.uniform(0, 5) + const
displacement2 = np.random.uniform(5, 10) - const

for step in range(5):
    wandb.log({
        "acc": 0.04 * (math.log(1 + step + random.random()) + random.random() * run.config.learning_rate + random.random() + displacement1 + random.random() * run.config.momentum),
        "val_acc": 0.04 * (math.log(1 + step + random.random()) + random.random() * run.config.learning_rate - random.random() + displacement1),
        "loss": 0.04 * (4 - math.log(1 + step + random.random()) + random.random() * run.config.momentum + random.random() + displacement2),
        "val_loss": 0.04 * (5 - math.log(1 + step + random.random()) + random.random() * run.config.learning_rate - random.random() + displacement2),
        "train_step": step*25
    })

#print(run.config.dataset)
#print(run.config.model)
