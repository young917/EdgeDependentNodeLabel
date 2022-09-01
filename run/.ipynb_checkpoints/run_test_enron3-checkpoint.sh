cd ..
cuda=3
seedlist=("0" "10" "100")
for seed in ${seedlist[@]}
do
    CUDA_VISIBLE_DEVICES=${cuda} python train_wdgl_w.py --dataset_name emailEnron --k 0 --embedder hnhn --num_layers 1 --scorer sm --scorer_num_layers 1 --optimizer adam --bs 64 --dropout 0.7 --gamma 0.99 --dim_hidden 64 --lr 0.001 --dim_edge 128 --dim_vertex 128 --epochs 100 --test_epoch 5 --sampling 100 --evaltype test --fix_seed --seed ${seed} --use_gpu
done