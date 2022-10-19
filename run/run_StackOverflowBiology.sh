cd ..
seedlist=("0" "10" "100")
for seed in ${seedlist[@]}
do
	# BaselineU and BaselineP
    python evaluate_baseline.py --dataset_name StackOverflowBiology --k 0 --fix_seed --seed ${seed} --evaltype test
	# HNHN
    python train.py --dataset_name StackOverflowBiology --embedder hnhn --num_layers 2 --scorer sm --scorer_num_layers 1 --bs 64 --lr 0.001 --sampling -1 --dropout 0.7 --optimizer "adam" --k 0 --gamma 0.99 --dim_hidden 64 --dim_edge 128 --dim_vertex 128 --epochs 100 --test_epoch 5 --evaltype test --save_epochs 1 --seed ${seed} --fix_seed
	# HGNN
    python train.py --dataset_name StackOverflowBiology --embedder hgnn --num_layers 1 --scorer sm --scorer_num_layers 1 --bs 128 --lr 0.001 --sampling -1 --dropout 0.7 --optimizer "adam" --k 0 --gamma 0.99 --dim_hidden 64 --dim_edge 128 --dim_vertex 128 --epochs 100 --test_epoch 5 --evaltype test --save_epochs 1 --seed ${seed} --fix_seed
	# HCHA
    python train_full_batch.py --dataset_name StackOverflowBiology --embedder hcha --num_layers 2 --scorer sm --scorer_num_layers 1 --lr 0.05 --sampling -1 --dropout 0.7 --optimizer "adam" --k 0 --gamma 0.99 --dim_hidden 64 --dim_edge 128 --dim_vertex 128 --epochs 100 --test_epoch 5 --evaltype test --save_epochs 1 --seed ${seed} --fix_seed
	# HAT
    python train.py --dataset_name StackOverflowBiology --embedder hat --num_layers 1 --scorer sm --scorer_num_layers 1 --bs 64 --lr 0.0001 --sampling -1 --dropout 0.7 --optimizer "adam" --k 0 --gamma 0.99 --dim_hidden 64 --dim_edge 128 --dim_vertex 128 --epochs 100 --test_epoch 5 --evaltype test --save_epochs 1 --seed ${seed} --fix_seed
	# UniGCNII
    python train.py --dataset_name StackOverflowBiology --embedder unigcnii --num_layers 2 --scorer sm --scorer_num_layers 1 --bs 128 --lr 0.001 --sampling -1 --dropout 0.7 --optimizer "adam" --k 0 --gamma 0.99 --dim_hidden 64 --dim_edge 128 --dim_vertex 128 --epochs 100 --test_epoch 5 --evaltype test --save_epochs 1 --seed ${seed} --fix_seed
	# HST
    python train.py --dataset_name StackOverflowBiology --embedder transformer --att_type_v pure --agg_type_v pure --att_type_e pure --agg_type_e pure --num_att_layer 2 --num_layers 1 --scorer sm --scorer_num_layers 1 --bs 128 --lr 0.001 --sampling 40 --dropout 0.7 --optimizer "adam" --k 0 --gamma 0.99 --dim_hidden 64 --dim_edge 128 --dim_vertex 128 --epochs 100 --test_epoch 5 --evaltype test --save_epochs 1 --seed ${seed} --fix_seed
	# AST
    python train.py --dataset_name StackOverflowBiology --embedder transformer --att_type_v NoAtt --agg_type_v pure2 --att_type_e NoAtt --agg_type_e pure2 --num_layers 2 --scorer sm --scorer_num_layers 1 --bs 64 --lr 0.001 --sampling 40 --dropout 0.7 --optimizer "adam" --k 0 --gamma 0.99 --dim_hidden 64 --dim_edge 128 --dim_vertex 128 --epochs 100 --test_epoch 5 --evaltype test --save_epochs 1 --seed ${seed} --fix_seed
	# WHATsNET
    python train.py --dataset_name StackOverflowBiology --vorder_input "degree_nodecentrality,eigenvec_nodecentrality,pagerank_nodecentrality,kcore_nodecentrality" --embedder transformer --att_type_v OrderPE --agg_type_v PrevQ --att_type_e OrderPE --agg_type_e PrevQ --num_att_layer 2 --num_layers 2 --scorer sm --scorer_num_layers 1 --bs 128 --lr 0.0001 --sampling 40 --dropout 0.7 --optimizer "adam" --k 0 --gamma 0.99 --dim_hidden 64 --dim_edge 128 --dim_vertex 128 --epochs 100 --test_epoch 5 --evaltype test --save_epochs 1 --seed ${seed} --fix_seed
	# WHATsNET w/o WithinOrderPE
    python train.py --dataset_name StackOverflowBiology --embedder transformer --att_type_v pure --agg_type_v PrevQ --att_type_e pure --agg_type_e PrevQ --num_att_layer 2 --num_layers 2 --scorer sm --scorer_num_layers 1 --bs 128 --lr 0.001 --sampling 40 --dropout 0.7 --optimizer "adam" --k 0 --gamma 0.99 --dim_hidden 64 --dim_edge 128 --dim_vertex 128 --epochs 100 --test_epoch 5 --evaltype test --save_epochs 1 --seed ${seed} --fix_seed
	# WHATsNET w/o WithinATT
    python train.py --dataset_name StackOverflowBiology --embedder transformer --att_type_v NoAtt --agg_type_v PrevQ --att_type_e NoAtt --agg_type_e PrevQ --num_layers 2 --scorer sm --scorer_num_layers 1 --bs 128 --lr 0.001 --sampling 40 --dropout 0.7 --optimizer "adam" --k 0 --gamma 0.99 --dim_hidden 64 --dim_edge 128 --dim_vertex 128 --epochs 100 --test_epoch 5 --evaltype test --save_epochs 1 --seed ${seed} --fix_seed
done