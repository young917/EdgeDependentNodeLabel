cd ..
cuda=3
sp=100
outdim=128
dim_hidden=64
num_inds=4
gamma=0.99
dropout=0.7

CUDA_VISIBLE_DEVICES=${cuda} python train_wdgl_w.py --dataset_name emailEu --output_dim 2 --embedder hnhn --num_layers 2 --scorer sm --scorer_num_layers 1 --optimizer "adam" --k 0 --bs 64 --dropout ${dropout} --gamma ${gamma} --dim_hidden ${dim_hidden} --lr 0.001 --dim_edge ${outdim} --dim_vertex ${outdim} --epochs 100 --test_epoch 5 --sampling ${sp} --use_gpu
CUDA_VISIBLE_DEVICES=${cuda} python train_wdgl_w.py --dataset_name emailEu --output_dim 2 --embedder hnhn --num_layers 2 --scorer sm --scorer_num_layers 1 --optimizer "adam" --k 0 --bs 128 --dropout ${dropout} --gamma ${gamma} --dim_hidden ${dim_hidden} --lr 0.0001 --dim_edge ${outdim} --dim_vertex ${outdim} --epochs 100 --test_epoch 5 --sampling ${sp} --use_gpu
