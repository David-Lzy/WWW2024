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
        swap=True,
        kl_loss=True,
        CW=True,
        train_method_path=train_method,
        sign_only=False,
        eps_init=0.01,
        )
    model.perturb_all(
        to_device=True,
        override=True,
    )
    attack_method = os.path.join(train_method, model.attack_method_path)
# p_file_name		parameter	sign	eps_init	 swap	 KL		CW
# ATTACK.5.Py		SWAP+CW		FALSE	0.01		True	True	True
concat_metrics_attack(method=attack_method, datasets=datasets)