cuda=0
cd ..
seedlist=("1000" "10000")
for seed in ${seedlist[@]}
do
    # BaselineU and BaselineP
	python evaluate_baseline.py --dataset_name emailEnron --k 0 --fix_seed --seed ${seed} --evaltype test
	# HNHN
    CUDA_VISIBLE_DEVICES=${cuda} python train.py --dataset_name emailEnron --embedder hnhn --num_layers 1 --scorer sm --scorer_num_layers 1 --bs 128 --lr 0.001 --sampling 100 --dropout 0.7 --optimizer "adam" --k 0 --gamma 0.99 --dim_hidden 64 --dim_edge 128 --dim_vertex 128 --epochs 100 --test_epoch 5 --evaltype test --save_epochs 1 --seed ${seed} --fix_seed
	# HGNN
    CUDA_VISIBLE_DEVICES=${cuda} python train.py --dataset_name emailEnron --embedder hgnn --num_layers 1 --scorer sm --scorer_num_layers 1 --bs 64 --lr 0.001 --sampling 40 --dropout 0.7 --optimizer "adam" --k 0 --gamma 0.99 --dim_hidden 64 --dim_edge 128 --dim_vertex 128 --epochs 100 --test_epoch 5 --evaltype test --save_epochs 1 --seed ${seed} --fix_seed
	# HCHA
    CUDA_VISIBLE_DEVICES=${cuda} python train_full_batch.py --dataset_name emailEnron --embedder hcha --num_layers 1 --scorer sm --scorer_num_layers 1 --lr 0.05 --sampling -1 --dropout 0.7 --optimizer "adam" --k 0 --gamma 0.99 --dim_hidden 64 --dim_edge 128 --dim_vertex 128 --epochs 100 --test_epoch 5 --evaltype test --save_epochs 1 --seed ${seed} --fix_seed
	# HAT
    CUDA_VISIBLE_DEVICES=${cuda} python train.py --dataset_name emailEnron --embedder hat --num_layers 1 --scorer sm --scorer_num_layers 1 --bs 128 --lr 0.001 --sampling 100 --dropout 0.7 --optimizer "adam" --k 0 --gamma 0.99 --dim_hidden 64 --dim_edge 128 --dim_vertex 128 --epochs 100 --test_epoch 5 --evaltype test --save_epochs 1 --seed ${seed} --fix_seed
	# UniGCNII
    CUDA_VISIBLE_DEVICES=${cuda} python train.py --dataset_name emailEnron --embedder unigcnii --num_layers 1 --scorer sm --scorer_num_layers 1 --bs 128 --lr 0.001 --sampling 100 --dropout 0.7 --optimizer "adam" --k 0 --gamma 0.99 --dim_hidden 64 --dim_edge 128 --dim_vertex 128 --epochs 100 --test_epoch 5 --evaltype test --save_epochs 1 --seed ${seed} --fix_seed
done