import wandb

run = wandb.init(project="stab-stab", name="william-shatner")
for i in range(10):
    run.log({"dead-babysitters": i})