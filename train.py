import wandb
import time
import random
import re
print("Running demo scriptasdasdasdasdasdas2")
config_dict = {
    "lr": 0.01,
    "decay": 1e-6,
    "epochs": 10,
    "language": "en",
    "model": "wandb-artifact://kylegoyette/triggers-demo/my-dataset:latest"
}


run = wandb.init(project="triggers-demo", config=config_dict)

# model = run.use_artifact("my-bad-model:latest")
# dataset = run.use_artifact("my-dataset:latest", use_as="dataset")

#run.config.model = "wandb-artifact://my-bad-model:latest"
# run.config.dataset = "wandb-artifact://my-dataset:latest"

# print("fork", run.config.model)
# print(f"Using model {run.config.model.name}")
# print()

#run.config.model = model
#run.config.dataset = dataset
# if "my-good-model" in run.config.model.name:
#     v1 = 2.0
# else:
    # v1 = 0.5
v1 = 1
for epoch in range(run.config.epochs):
    time.sleep(2)
    wandb.log({"good_metric": random.random()*10*run.config.lr*v1*epoch + run.config.decay*1e6})
    wandb.log({"bad_metric": 1 - epoch*(random.random()*10*run.config.lr*v1*epoch + run.config.decay*1e6)})