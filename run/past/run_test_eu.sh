CUDA_VISIBLE_DEVICES=0 python train_wdgl_w.py --dataset_name emailEu --output_dim 2 --k 0 --embedder transformer --encode_type pure --decode_type pure --num_layers 1 --scorer sm --scorer_num_layers 1 --optimizer adam --bs 64 --dropout 0.7 --gamma 0.99 --dim_hidden 64 --lr 0.0001 --dim_edge 128 --dim_vertex 128 --epochs 100 --test_epoch 5 --sampling 40 --evaltype test --fix_seed --seed 100 --use_gpu

# cd ..
# cuda=0
# seedlist=("10" "100")
# for seed in ${seedlist[@]}
# do
#     CUDA_VISIBLE_DEVICES=${cuda} python train_wdgl_w.py --dataset_name emailEu --output_dim 2 --k 0 --embedder transformer --decode_type pure2 --num_layers 1 --scorer sm --scorer_num_layers 1 --optimizer adam --bs 128 --dropout 0.7 --gamma 0.99 --dim_hidden 64 --lr 0.001 --dim_edge 128 --dim_vertex 128 --epochs 100 --test_epoch 5 --sampling 40 --evaltype test --fix_seed --seed ${seed} --use_gpu
#     CUDA_VISIBLE_DEVICES=${cuda} python train_wdgl_w.py --dataset_name emailEu --output_dim 2 --k 0 --embedder hat --num_layers 1 --scorer sm --scorer_num_layers 1 --optimizer adam --bs 64 --dropout 0.7 --gamma 0.99 --dim_hidden 64 --lr 0.001 --dim_edge 128 --dim_vertex 128 --epochs 100 --test_epoch 5 --sampling 40 --evaltype test --fix_seed --seed ${seed} --use_gpu
#     CUDA_VISIBLE_DEVICES=${cuda} python train_wdgl_w.py --dataset_name emailEu --output_dim 2 --k 0 --embedder hnhn --num_layers 1 --scorer sm --scorer_num_layers 1 --optimizer adam --bs 64 --dropout 0.7 --gamma 0.99 --dim_hidden 64 --lr 0.001 --dim_edge 128 --dim_vertex 128 --epochs 100 --test_epoch 5 --sampling 100 --evaltype test --fix_seed --seed ${seed} --use_gpu
# done