python main.py --factorization_weight 0.99 --regression_weight 0.01 --logdir run/shared=True_LF=0.99_LR=0.01 
python main.py --factorization_weight 0.5 --regression_weight 0.5 --logdir run/shared=True_LF=0.5_LR=0.5 
python main.py --no_shared_embeddings --factorization_weight 0.99 --regression_weight 0.01 --logdir run/shared=False_LF=0.99_LR=0.01 
python main.py --no_shared_embeddings --factorization_weight 0.5 --regression_weight 0.5 --logdir run/shared=False_LF=0.5_LR=0.5