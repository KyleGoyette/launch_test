import wandb
# from wandb.sdk.launch.launch_add import launch_add
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

random.seed(run.config.seed)
artifact = wandb.Artifact('my-orig-dataset', type='dataset')
artifact.add_file('my-dataset.txt')
run.log_artifact(artifact)
#dataset = run.use_artifact('my-dataset2:v0')
#run.config.dataset = dataset
#print(dataset._name)
# if dataset._name == 'my-dataset2':
#     const = 1
# else:
#     const = 0

# for step in range(5):
#     wandb.log({
#         "acc": 0.04 * (math.log(1 + step + random.random()) + random.random() * run.config.learning_rate + random.random() + displacement1 + random.random() * run.config.momentum),
#         "val_acc": 0.04 * (math.log(1 + step + random.random()) + random.random() * run.config.learning_rate - random.random() + displacement1),
#         "loss": 0.04 * (4 - math.log(1 + step + random.random()) + random.random() * run.config.momentum + random.random() + displacement2),
#         "val_loss": 0.04 * (5 - math.log(1 + step + random.random()) + random.random() * run.config.learning_rate - random.random() + displacement2),
#         "train_step": step*25
#     })
