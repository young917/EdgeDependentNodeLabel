cuda=1
cd ..
seedlist=("10000" "500")
for seed in ${seedlist[@]}
do
	# HAT
    CUDA_VISIBLE_DEVICES=${cuda} python train.py --dataset_name AMinerAuthor --embedder hat --num_layers 1 --scorer sm --scorer_num_layers 1 --bs 512 --lr 0.001 --sampling 40 --dropout 0.7 --optimizer "adam" --k 0 --gamma 0.99 --dim_hidden 64 --dim_edge 128 --dim_vertex 128 --epochs 100 --test_epoch 5 --evaltype test --save_epochs 1 --seed ${seed} --fix_seed --use_gpu
	# UniGCNII
    CUDA_VISIBLE_DEVICES=${cuda} python train.py --dataset_name AMinerAuthor --embedder unigcnii --num_layers 2 --scorer sm --scorer_num_layers 1 --bs 512 --lr 0.001 --sampling 40 --dropout 0.7 --optimizer "adam" --k 0 --gamma 0.99 --dim_hidden 64 --dim_edge 128 --dim_vertex 128 --epochs 100 --test_epoch 5 --evaltype test --save_epochs 1 --seed ${seed} --fix_seed --use_gpu
done