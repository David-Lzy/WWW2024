#!/bin/bash

# If CW=False, paramater c will be ignored
# IF swap=False, paramater swap_index will be ignored


python headless_attack.py --batch_size 128 --epoch 1000 --c 1e-5 --swap True --kl_loss True --CW False  --swap_index 1 --override True

python headless_attack.py --batch_size 128 --epoch 1000 --c 1e-5 --swap True --kl_loss False --CW False  --swap_index 1 --override True

python headless_attack.py --batch_size 128 --epoch 1000 --c 1e-5 --swap True --kl_loss True --CW True  --swap_index 1 --override True

python headless_attack.py --batch_size 128 --epoch 1000 --c 1e-5 --swap True --kl_loss False --CW True  --swap_index 1 --override True

python headless_attack.py --batch_size 128 --epoch 1000 --c 1e-5 --swap False --kl_loss True --CW False  --swap_index 1 --override True

python headless_attack.py --batch_size 128 --epoch 1000 --c 1e-5 --swap False --kl_loss False --CW False  --swap_index 1 --override True

python headless_attack.py --batch_size 128 --epoch 1000 --c 1e-5 --swap False --kl_loss True --CW True  --swap_index 1 --override True

python headless_attack.py --batch_size 128 --epoch 1000 --c 1e-5 --swap False --kl_loss False --CW True  --swap_index 1 --override True
