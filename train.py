import wandb
import random
import time

config_dict = {
    "param1": 0.01,
    "param2": 1,
}
wandb.init(project="launch-demo", config=config_dict)
config = wandb.config

for step in range (30):
    wandb.log({"metric1": random.random() * config.param1 + config.param2})
    time.sleep(1)