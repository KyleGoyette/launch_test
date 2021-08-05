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
artifact = wandb.Artifact('my-dataset', type='dataset')
artifact.add_file('my-dataset.txt')
run.log_artifact(artifact)
run.config.dataset = artifact
for step in range(run.config.epochs):
    wandb.log({
        "acc": 0.04 * (math.log(1 + step + random.random()) + random.random() * run.config.learning_rate + random.random() + displacement1 + random.random() * run.config.momentum),
        "val_acc": 0.04 * (math.log(1 + step + random.random()) + random.random() * run.config.learning_rate - random.random() + displacement1),
        "loss": 0.04 * (4 - math.log(1 + step + random.random()) + random.random() * run.config.momentum + random.random() + displacement2),
        "val_loss": 0.04 * (5 - math.log(1 + step + random.random()) + random.random() * run.config.learning_rate - random.random() + displacement2),
    })


# queued_job.wait_until_running()
# run = queued_job.run
# run.wait_until_finished()

# # import wandb
# # from wandb.sdk.launch.launch_add import launch_add
# jobs = []
# for i in range(1):
#     queued_job = launch_add(uri="https://wandb.ai/kylegoyette/launch-demo/runs/21g96em6", config={
#   "uri": "https://wandb.ai/kylegoyette/launch-demo/runs/21g96em6",
#   "project": "cvpsync",
#   "entity": "kylegoyette",
#   "overrides": {
#     "run_config": {
#       "learning_rate": 999
#     }
#   }
# })
#     jobs.append(queued_job)
# for queued_job in jobs:
#     queued_job.wait_until_running()
#     run = queued_job.run
#     run.wait_until_finished()