import wandb
from wandb.sdk.launch.launch_add import launch_add
import argparse
import math
import time
import random

parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=0.1)
parser.add_argument("--epochs", type=int, default=15)
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
    time.sleep(2)
    wandb.log({
        "acc": 0.04 * (math.log(1 + step + random.random()) + random.random() * run.config.learning_rate + random.random() + displacement1 + random.random() * run.config.momentum),
        "val_acc": 0.04 * (math.log(1 + step + random.random()) + random.random() * run.config.learning_rate - random.random() + displacement1),
        "loss": 0.04 * (4 - math.log(1 + step + random.random()) + random.random() * run.config.momentum + random.random() + displacement2),
        "val_loss": 0.04 * (5 - math.log(1 + step + random.random()) + random.random() * run.config.learning_rate - random.random() + displacement2),
    })

# queued_job = launch_add(uri="https://wandb.ai/kylegoyette/launch-demo/runs/21g96em6")
# queued_job.wait_until_running()
# run = queued_job.run
# run.wait_until_finished()
