import argparse
import wandb

parser = argparse.ArgumentParser()
parser.description = 'Transform an artifact'
parser.add_argument('--input_num', type=float, required=True)

args = parser.parse_args()
with wandb.init(config=args, project='shawn_add') as run:
    run.summary['output'] = run.config.input_num + 3