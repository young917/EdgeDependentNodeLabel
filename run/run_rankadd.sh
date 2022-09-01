cd ..
cuda=1
lrlist=("0.001" "0.0001")
numlayerlist=("1" "2")
numattlayerlist=("1" "2")
bslist=("64" "128" "32" "16")
splist=("-1")
outdim=128
dim_hidden=64
num_inds=4
gamma=0.99
dropout=0.7

for bs in ${bslist[@]}
do
    for num_att_layers in ${numattlayerlist[@]}
    do
        for num_layers in ${numlayerlist[@]}
        do
            for lr in ${lrlist[@]}
            do
                for sp in ${splist[@]}
                do
                    CUDA_VISIBLE_DEVICES=${cuda} python train_wdgl_w.py --dataset_name DBLP2 --exist_hedgename --vrank_input degree_nodecentrality,eigenvec_nodecentrality,pagerank_nodecentrality,kcore_nodecentrality --embedder transformer --att_type_v RankAdd --agg_type_v PrevQ --att_type_e pure --agg_type_e PrevQ --num_att_layer ${num_att_layers} --num_layers ${num_layers} --scorer sm --scorer_num_layers 1 --optimizer "adam" --k 0 --bs ${bs} --dropout ${dropout} --gamma ${gamma} --dim_hidden ${dim_hidden} --lr ${lr} --dim_edge ${outdim} --dim_vertex ${outdim} --epochs 100 --test_epoch 5 --sampling ${sp} --use_gpu
                done 
            done
        done
    done
done