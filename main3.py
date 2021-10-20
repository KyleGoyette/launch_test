import wandb
import random

config_dict = {
    "lr": 0.01,
    "decay": 1e-6,
    "epochs": 10,
}
run = wandb.init(project="launch-artifact-demo", config=config_dict)

dataset = run.use_artifact("my-dataset:latest", use_as="dataset")
model = run.use_artifact("my-bad-model:latest", use_as="model")

print(f"Using model {model.name}")
print(f"Using dataset: {dataset.name}")
run.config.dataset = dataset
for epoch in range(run.config.epochs):
    if model.name == "my-good-model":
        v1 = 1.0
    else:
        v1 = 0.5
    wandb.log({"good_metric": epoch*(random.random() * run.config.lr*v1 + run.config.decay*1e6)})
    wandb.log({"bad_metric": 1 - epoch*(random.random() * run.config.lr*v1 + run.config.decay*1e6)})