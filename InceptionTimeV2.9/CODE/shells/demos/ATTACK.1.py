import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.dirname(__file__))
from Package import *
logger = logging.getLogger()
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)
from CODE.attack.mix import Mix


model_class = Mix
train_method = 'defence=None'
attack_method = ''

for dataset in datasets:
    model = model_class(
        dataset=dataset,
        batch_size=128,
        epoch=1000,
        swap=False,
        kl_loss=False,
        CW=False,
        train_method_path=train_method,
        sign_only=True,
        alpha=0.0005,
        eps_init=0,
        )
    model.perturb_all(
        to_device=True,
        override=True,
    )
    attack_method = os.path.join(train_method, model.attack_method_path)
# p_file_name		parameter	sign	eps_init	 swap	 KL		CW
# ATTACK.1.Py		BIM			TRUE	0			FALSE	FALSE	FALSE
concat_metrics_attack(method=attack_method, datasets=datasets)