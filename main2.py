import wandb
import numpy as np
import time

run = wandb.init(project="triggers-demo")

artifact = wandb.Artifact('my-dataset', type='dataset')
print(artifact.name)
artifact.add_file('my-dataset.txt')
run.log_artifact(artifact)
artifact.wait()
#print(artifact._sequence_name)
run.config.dataset = artifact
run.log_code()
#wandb.log({"stuff": 'asdf'})