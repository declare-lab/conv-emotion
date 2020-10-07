import sys
import os
import argparse

sys.path.append(os.getcwd())

parser = argparse.ArgumentParser()
parser.add_argument("--experiment_type", type=str, default='atomic',
                    choices=["atomic", "conceptnet"])
parser.add_argument("--experiment_num", type=str, default="0")

args = parser.parse_args()

if args.experiment_type == "atomic":
    from main_atomic import main
    main(args.experiment_num)
if args.experiment_type == "conceptnet":
    from main_conceptnet import main
    main(args.experiment_num)
