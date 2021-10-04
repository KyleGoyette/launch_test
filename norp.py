
import wandb
run  = wandb.init(project="test-id", resume="auto")
step = run.step
wandb.log({"logged":1234})

wandb.log({"second_cell":step})




