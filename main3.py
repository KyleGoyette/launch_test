import wandb
import time
import random
import re
print("Running demo script")
config_dict = {
    "lr": 0.01,
    "decay": 1e-6,
    "epochs": 10,
}


run = wandb.init(project="triggers-demo", config=config_dict)

model = run.use_artifact("my-bad-model:latest", use_as="model")
dataset = run.use_artifact("my-dataset:latest", use_as="dataset")
print()
print(f"Using model {model.name}")
print()

run.config.model = model
run.config.dataset = dataset
if "my-good-model" in model.name:
    v1 = 2.0
else:
    v1 = 0.5
for epoch in range(run.config.epochs):
    wandb.log({"good_metric": random.random()*10*run.config.lr*v1*epoch + run.config.decay*1e6})
    wandb.log({"bad_metric": 1 - epoch*(random.random()*10*run.config.lr*v1*epoch + run.config.decay*1e6)})