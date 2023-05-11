cd ../..
cuda=0
seedlist=("0" "10" "100" "500" "10000")

for seed in ${seedlist[@]}
do
    # WHATsNET
    CUDA_VISIBLE_DEVICES=${cuda} python train.py --dataset_name Etail  --inputdir downstreamdata/ --vorder_input "degree_nodecentrality,eigenvec_nodecentrality,pagerank_nodecentrality,kcore_nodecentrality" --embedder transformer --att_type_v OrderPE --agg_type_v PrevQ --att_type_e OrderPE --agg_type_e PrevQ --num_att_layer 2 --num_layers 1 --scorer sm --scorer_num_layers 1 --bs 64 --lr 0.001 --sampling -1 --dropout 0.7 --optimizer "adam" --k 0 --gamma 0.99 --dim_hidden 64 --dim_edge 128 --dim_vertex 128 --epochs 300 --test_epoch 2 --evaltype test --save_epochs 1 --seed ${seed} --fix_seed
    CUDA_VISIBLE_DEVICES=${cuda} python predict.py --dataset_name Etail  --inputdir downstreamdata/ --vorder_input "degree_nodecentrality,eigenvec_nodecentrality,pagerank_nodecentrality,kcore_nodecentrality" --embedder transformer --att_type_v OrderPE --agg_type_v PrevQ --att_type_e OrderPE --agg_type_e PrevQ --num_att_layer 2 --num_layers 1 --scorer sm --scorer_num_layers 1 --bs 64 --lr 0.001 --sampling -1 --dropout 0.7 --optimizer "adam" --k 0 --gamma 0.99 --dim_hidden 64 --dim_edge 128 --dim_vertex 128 --epochs 300 --test_epoch 2 --evaltype test --save_epochs 1 --seed ${seed} --fix_seed --outputname our
    # AST
    CUDA_VISIBLE_DEVICES=${cuda} python train.py --dataset_name Etail  --inputdir downstreamdata/ --embedder transformer --att_type_v NoAtt --agg_type_v pure2 --att_type_e NoAtt --agg_type_e pure2 --num_layers 1 --scorer sm --scorer_num_layers 1 --bs 128 --lr 0.05 --sampling -1 --dropout 0.7 --optimizer "adam" --k 0 --gamma 0.99 --dim_hidden 64 --dim_edge 128 --dim_vertex 128 --epochs 300 --test_epoch 2 --evaltype test --save_epochs 1 --seed ${seed} --fix_seed 
    CUDA_VISIBLE_DEVICES=${cuda} python predict.py --dataset_name Etail  --inputdir downstreamdata/ --embedder transformer --att_type_v NoAtt --agg_type_v pure2 --att_type_e NoAtt --agg_type_e pure2 --num_layers 1 --scorer sm --scorer_num_layers 1 --bs 128 --lr 0.05 --sampling -1 --dropout 0.7 --optimizer "adam" --k 0 --gamma 0.99 --dim_hidden 64 --dim_edge 128 --dim_vertex 128 --epochs 300 --test_epoch 2 --evaltype test --save_epochs 1 --seed ${seed} --fix_seed --outputname ast 
    # HST
    CUDA_VISIBLE_DEVICES=${cuda} python train.py --dataset_name Etail  --inputdir downstreamdata/ --embedder transformer --att_type_v pure --agg_type_v pure --att_type_e pure --agg_type_e pure --num_att_layer 2 --num_layers 1 --scorer sm --scorer_num_layers 1 --bs 512 --lr 0.003 --sampling -1 --dropout 0.7 --optimizer "adam" --k 0 --gamma 0.99 --dim_hidden 64 --dim_edge 128 --dim_vertex 128 --epochs 300 --test_epoch 2 --evaltype test --save_epochs 1 --seed ${seed} --fix_seed
    CUDA_VISIBLE_DEVICES=${cuda} python predict.py --dataset_name Etail  --inputdir downstreamdata/ --embedder transformer --att_type_v pure --agg_type_v pure --att_type_e pure --agg_type_e pure --num_att_layer 2 --num_layers 1 --scorer sm --scorer_num_layers 1 --bs 512 --lr 0.003 --sampling -1 --dropout 0.7 --optimizer "adam" --k 0 --gamma 0.99 --dim_hidden 64 --dim_edge 128 --dim_vertex 128 --epochs 300 --test_epoch 2 --evaltype test --save_epochs 1 --seed ${seed} --fix_seed --outputname hst
done

cd ProductReturnPred
for seed in ${seedlist[@]}
do
    cd makedata
    python prepare_predicted.py --outputname our_${seed}
    python prepare_predicted.py --outputname ast_${seed}
    python prepare_predicted.py --outputname hst_${seed}
    cd ../script
    python main_prod.py --model our_${seed}
    python main_prod.py --model ast_${seed}
    python main_prod.py --model hst_${seed}
    cd ..
done
