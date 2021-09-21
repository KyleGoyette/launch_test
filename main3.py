import wandb
import time
run = wandb.init(project="arti-test")
#time.sleep(15)
run.use_artifact('my-dataset2:latest')
#run.use_artifact('arti-test/my-dataset2:latest')

wandb.log({"stuff": 'asdf'})