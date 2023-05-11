cd ../..
cuda=0

seedlist=("0" "10" "100" "500" "10000")
for seed in ${seedlist[@]}
do
    # WHATsNET
    CUDA_VISIBLE_DEVICES=${cuda} python train.py --dataset_name AMiner_rank --exist_hedgename --inputdir downstreamdata/ --vorder_input "degree_nodecentrality,eigenvec_nodecentrality,pagerank_nodecentrality,kcore_nodecentrality" --embedder transformer --att_type_v OrderPE --agg_type_v PrevQ --att_type_e OrderPE --agg_type_e PrevQ --num_att_layer 2 --num_layers 2 --scorer sm --scorer_num_layers 1 --bs 32 --lr 0.0005 --sampling -1 --dropout 0.7 --optimizer "adam" --k 10000 --gamma 0.99 --dim_hidden 64 --dim_edge 128 --dim_vertex 128 --epochs 300 --test_epoch 2 --test_epoch 2 --evaltype test --save_epochs 1 --seed ${seed} --fix_seed
    CUDA_VISIBLE_DEVICES=${cuda} python predict.py --dataset_name AMiner_rank --exist_hedgename --inputdir downstreamdata/ --vorder_input "degree_nodecentrality,eigenvec_nodecentrality,pagerank_nodecentrality,kcore_nodecentrality" --embedder transformer --att_type_v OrderPE --agg_type_v PrevQ --att_type_e OrderPE --agg_type_e PrevQ --num_att_layer 2 --num_layers 2 --scorer sm --scorer_num_layers 1 --bs 32 --lr 0.0005 --sampling -1 --dropout 0.7 --optimizer "adam" --k 10000 --gamma 0.99 --dim_hidden 64 --dim_edge 128 --dim_vertex 128 --epochs 300 --test_epoch 2 --test_epoch 2 --evaltype test --save_epochs 1 --seed ${seed} --fix_seed --outputname our
    # AST
    CUDA_VISIBLE_DEVICES=${cuda} python train.py --dataset_name AMiner_rank --exist_hedgename --inputdir downstreamdata/ --embedder transformer --att_type_v NoAtt --agg_type_v pure2 --att_type_e NoAtt --agg_type_e pure2 --num_layers 1 --scorer sm --scorer_num_layers 1 --bs 32 --lr 0.0005 --sampling -1 --dropout 0.7 --optimizer "adam" --k 10000 --gamma 0.99 --dim_hidden 64 --dim_edge 128 --dim_vertex 128 --epochs 300 --test_epoch 2 --evaltype test --save_epochs 1 --seed ${seed} --fix_seed
    CUDA_VISIBLE_DEVICES=${cuda} python predict.py --dataset_name AMiner_rank --exist_hedgename --inputdir downstreamdata/ --embedder transformer --att_type_v NoAtt --agg_type_v pure2 --att_type_e NoAtt --agg_type_e pure2 --num_layers 1 --scorer sm --scorer_num_layers 1 --bs 32 --lr 0.0005 --sampling -1 --dropout 0.7 --optimizer "adam" --k 10000 --gamma 0.99 --dim_hidden 64 --dim_edge 128 --dim_vertex 128 --epochs 300 --test_epoch 2 --evaltype test --save_epochs 1 --seed ${seed} --fix_seed --outputname ast
    # HST
    CUDA_VISIBLE_DEVICES=${cuda} python train.py --dataset_name AMiner_rank --exist_hedgename --inputdir downstreamdata/ --embedder transformer --att_type_v pure --agg_type_v pure --att_type_e pure --agg_type_e pure --num_att_layer 2 --num_layers 1 --scorer sm --scorer_num_layers 1 --bs 32 --lr 0.0005 --sampling -1 --dropout 0.7 --optimizer "adam" --k 10000 --gamma 0.99 --dim_hidden 64 --dim_edge 128 --dim_vertex 128 --epochs 300 --test_epoch 2 --evaltype test --save_epochs 1 --seed ${seed} --fix_seed
    CUDA_VISIBLE_DEVICES=${cuda} python predict.py --dataset_name AMiner_rank --exist_hedgename --inputdir downstreamdata/ --embedder transformer --att_type_v pure --agg_type_v pure --att_type_e pure --agg_type_e pure --num_att_layer 2 --num_layers 1 --scorer sm --scorer_num_layers 1 --bs 32 --lr 0.0005 --sampling -1 --dropout 0.7 --optimizer "adam" --k 10000 --gamma 0.99 --dim_hidden 64 --dim_edge 128 --dim_vertex 128 --epochs 300 --test_epoch 2 --evaltype test --save_epochs 1 --seed ${seed} --fix_seed --outputname hst
done

cd Clustering
for seed in ${seedlist[@]}
do
    # our
    python clustering_aminer.py --predict_path our_${seed}
    python clustering_aminer.py --predict_path hst_${seed}
    python clustering_aminer.py --predict_path ast_${seed}
done

cd RankingAggregation
for seed in ${seedlist[@]}
do
    # our
    python aminer_ranking.py --outputpath our_${seed}
    python aminer_ranking.py --outputpath hst_${seed}
    python aminer_ranking.py --outputpath ast_${seed}
done
