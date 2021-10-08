import wandb
import time
run = wandb.init(project="arti-test")
time.sleep(5)
artifact = run.use_artifact('my-new-arti:latest', use_as="fork1")
artifact2 = run.use_artifact('my-new-arti:latest', use_as="fork2")
artifact3 = run.use_artifact('my-new-arti:latest')
artifact4 = run.use_artifact('my-new-arti:latest')
#artifact = wandb.Artifact('my-boom-arti', type='model', use_as="blah")
#run.use_artifact(artifact)
#artifact.wait()
print(artifact.version)
run.config.update({"arti": {"dataset": artifact}})

run.config.garbage = artifact2

print("ABJKDSA")
print(run.config.arti["dataset"])
print(run.config.garbage)
# artifact2 = run.use_artifact('my-team-dataset2:v0', use_as="worst-dataset")
# artifact3 = run.use_artifact('my-team-model:v0', use_as="model")
# run.config.dataset = artifact
# run.config.bad_dataset = artifact2
# run.config.model = artifact3
