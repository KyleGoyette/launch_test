import wandb
import time
run = wandb.init(project="arti-test")
time.sleep(5)
artifact = run.use_artifact('my-team-dataset2:v0', use_as="dataset")
artifact2 = run.use_artifact('my-team-dataset2:v0', use_as="worst-dataset")
artifact3 = run.use_artifact('my-team-model:v0', use_as="model")
run.config.dataset = artifact
run.config.bad_dataset = artifact2
run.config.model = artifact3
