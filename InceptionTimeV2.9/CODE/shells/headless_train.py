import sys
import os
import argparse

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.dirname(__file__))
from Package import *

logger = logging.getLogger()
logger.setLevel(logging.INFO)
handler.setFormatter(formatter)
logger.addHandler(handler)


def s_2_b(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ("yes", "true", "y", "t"):
        return True
    elif value.lower() in ("no", "false", "n", "f"):
        return False
    else:
        raise argparse.ArgumentTypeError(f"Invalid value for boolean argument: {value}")


parser = argparse.ArgumentParser(description="Run Trainer Demo")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
parser.add_argument("--epoch", type=int, default=750, help="Number of epochs")
parser.add_argument(
    "--override", type=s_2_b, default=False, help="Override existing results"
)
parser.add_argument(
    "--angle", type=s_2_b, default=False, help="If use angle in lossfuntion"
)

args = parser.parse_args()

for dataset in datasets:
    trainer = Trainer(
        dataset,
        batch_size=args.batch_size,
        epoch=args.epoch,
        override=args.override,
        angle=args.angle
    )
    trainer.train_and_evaluate()


concat_metrics_train(mode="train")
concat_metrics_train(mode="test")
