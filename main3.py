import wandb
import time
run = wandb.init(project="arti-test")
#time.sleep(15)
artifact = run.use_artifact('my-dataset2:v0')
#run.use_artifact('arti-test/my-dataset2:latest')
print(artifact.name)
wandb.log({"stuff": 'asdf'})