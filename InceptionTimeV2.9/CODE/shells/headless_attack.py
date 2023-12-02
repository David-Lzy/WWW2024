import sys
import os
import argparse

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.dirname(__file__))
from Package import *
from Package import ATTACK_OUTPUT_path
from CODE.attack.mix import Mix

model_class = Mix


# Create ArgumentParser object
parser = argparse.ArgumentParser(description="Your script description.")


def s_2_b(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ("yes", "true", "y", "t"):
        return True
    elif value.lower() in ("no", "false", "n", "f"):
        return False
    else:
        raise argparse.ArgumentTypeError(f"Invalid value for boolean argument: {value}")


# Add required parameters
parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
parser.add_argument("--epoch", type=int, default=1000, help="Number of epochs")
parser.add_argument("--c", type=float, default=1e-4, help="Value of c")
parser.add_argument("--swap", type=s_2_b, default=False, help="Swap flag")
parser.add_argument("--kl_loss", type=s_2_b, default=False, help="KL loss flag")
parser.add_argument("--swap_index", type=int, default=1, help="Swap index")
parser.add_argument("--CW", type=s_2_b, default=False, help="CW flag")
parser.add_argument("--override", type=s_2_b, default=False, help="override")
parser.add_argument("--to_device", type=s_2_b, default=False, help="to_device")
parser.add_argument("--angle", type=s_2_b, default=False, help="If use angle in lossfuntion"
)

# Parse command line parameters
args = parser.parse_args()


logger = logging.getLogger()
logger.setLevel(logging.INFO)
handler.setFormatter(formatter)
logger.addHandler(handler)


#Add your main code logic here
# For example, 
# you can call model_class and model.perturb_all with the above parameters
# Use parameters here to execute your code logic
logging.info(f"Batch Size: {args.batch_size}")
logging.info(f"Epoch: {args.epoch}")
logging.info(f"c: {args.c}")
logging.info(f"Swap: {args.swap}")
logging.info(f"KL Loss: {args.kl_loss}")
logging.info(f"Swap Index: {args.swap_index}")
logging.info(f"CW: {args.CW}")


if __name__ == "__main__":
    method = ""
    for i in datasets:
        model = model_class(
            dataset=i,
            batch_size=args.batch_size,
            epoch=args.epoch,
            c=args.c,
            swap=args.swap,
            kl_loss=args.kl_loss,
            swap_index=args.swap_index,
            CW=args.CW,
            angle=args.angle
        )
        method = model.method
        model.perturb_all(
            to_device=args.to_device,
            override=args.override,
        )

    concat_metrics_attack(method=method, datasets=datasets)
