import wandb
run = wandb.init(project="arti-test")
artifact = run.use_artifact('my-dataset:v0')
artifact = run.use_artifact('my-dataset:v0',use_as="blah")
run.config.dataset = artifact

wandb.log({"stuff": 'asdf'})