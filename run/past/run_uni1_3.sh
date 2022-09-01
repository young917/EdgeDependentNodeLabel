cd ..
cuda=2
lrlist=("0.001")
numlayerlist=("2")
bslist=("64")
splist=("40")
outdim=128
dim_hidden=64
num_inds=4
gamma=0.99
dropout=0.7

for num_layers in ${numlayerlist[@]}
do
    for lr in ${lrlist[@]}
    do    
        for bs in ${bslist[@]}
        do
            for sp in ${splist[@]}
            do
                CUDA_VISIBLE_DEVICES=${cuda} python train_wdgl_w.py --dataset_name emailEnron --embedder unigcnii --num_layers ${num_layers} --scorer sm --scorer_num_layers 1 --optimizer "adam" --k 0 --bs ${bs} --dropout ${dropout} --gamma ${gamma} --dim_hidden ${dim_hidden} --lr ${lr} --dim_edge ${outdim} --dim_vertex ${outdim} --epochs 100 --test_epoch 5 --sampling ${sp} --use_gpu
            done
        done
    done
done