#! bin/bash
for K in 1 2 4 6 8 10
do
python protonet.py --log_dir  --checkpoint_step 4500 --test --num_support $K --num_query 10
python maml.py --log_dir --checkpoint_step --test  --num_support $K  --num_query 10
done