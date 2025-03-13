#! bin/bash
for num_support in 5 1
do
python protonet.py  --num_way 5 --num_support $num_support
done