#! bin/bash
#1
python maml.py  --num_support 1 --num_way 5  --num_inner_steps 1  --inner_lr 0.4 --num_query 15
#3
python maml.py  --num_support 1 --num_way 5  --num_inner_steps 1  --inner_lr 0.04 --num_query 15
#4
python maml.py  --num_support 1 --num_way 5 --num_inner_steps 5  --inner_lr 0.04 --num_query 15
#5
python maml.py  --num_support 1 --num_way 5  --num_inner_steps 1  --inner_lr 0.4 --num_query 15  --learn_inner_lrs

