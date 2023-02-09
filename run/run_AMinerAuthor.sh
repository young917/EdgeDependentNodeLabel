cd ..
seedlist=("0" "10" "100" "500" "10000")
for seed in ${seedlist[@]}
do
    # BaselineU and BaselineP
    python evaluate_baseline.py --dataset_name AMinerAuthor --k 0 --fix_seed --seed ${seed} --evaltype test
    # HNHN
    python train.py --dataset_name AMinerAuthor --embedder hnhn --num_layers 1 --scorer sm --scorer_num_layers 1 --bs 256 --lr 0.001 --sampling 100 --dropout 0.7 --optimizer "adam" --k 0 --gamma 0.99 --dim_hidden 64 --dim_edge 128 --dim_vertex 128 --epochs 100 --test_epoch 5 --evaltype test --save_epochs 1 --seed ${seed} --fix_seed
    # HGNN
    python train.py --dataset_name AMinerAuthor --embedder hgnn --num_layers 2 --scorer sm --scorer_num_layers 1 --bs 256 --lr 0.001 --sampling 40 --dropout 0.7 --optimizer "adam" --k 0 --gamma 0.99 --dim_hidden 64 --dim_edge 128 --dim_vertex 128 --epochs 100 --test_epoch 5 --evaltype test --save_epochs 1 --seed ${seed} --fix_seed
    # HAT
    python train.py --dataset_name AMinerAuthor --embedder hat --num_layers 1 --scorer sm --scorer_num_layers 1 --bs 512 --lr 0.001 --sampling 40 --dropout 0.7 --optimizer "adam" --k 0 --gamma 0.99 --dim_hidden 64 --dim_edge 128 --dim_vertex 128 --epochs 100 --test_epoch 5 --evaltype test --save_epochs 1 --seed ${seed} --fix_seed
    # HCHA
    python train_full_batch.py --dataset_name AMinerAuthor --embedder hcha --num_layers 1 --scorer sm --scorer_num_layers 1 --lr 0.1 --sampling -1 --dropout 0.3 --optimizer "adam" --k 0 --gamma 0.99 --dim_hidden 64 --dim_edge 128 --dim_vertex 128 --epochs 300 --test_epoch 5 --evaltype test --seed ${seed} --fix_seed --save_epochs 1
    # UniGCNII
    python train.py --dataset_name AMinerAuthor --embedder unigcnii --num_layers 2 --scorer sm --scorer_num_layers 1 --bs 512 --lr 0.001 --sampling 40 --dropout 0.7 --optimizer "adam" --k 0 --gamma 0.99 --dim_hidden 64 --dim_edge 128 --dim_vertex 128 --epochs 100 --test_epoch 5 --evaltype test --save_epochs 1 --seed ${seed} --fix_seed
    # HNN
    python train_full_batch.py --dataset_name AMinerAuthor --embedder hnn --psi_num_layers 1 --efeat avg --num_layers 1 --scorer sm --scorer_num_layers 1 --lr 0.03 --sampling -1 --dropout 0.3 --optimizer "adam" --k 0 --gamma 0.99 --dim_hidden 64 --dim_edge 128 --dim_vertex 128 --epochs 300 --test_epoch 5 --evaltype test --seed ${seed} --fix_seed --save_epochs 1
    # HST
    python train.py --dataset_name AMinerAuthor --embedder whatsnet --att_type_v pure --agg_type_v pure --att_type_e pure --agg_type_e pure --num_att_layer 2 --num_layers 1 --scorer sm --scorer_num_layers 1 --bs 512 --lr 0.0001 --sampling 40 --dropout 0.7 --optimizer "adam" --k 0 --gamma 0.99 --dim_hidden 64 --dim_edge 128 --dim_vertex 128 --epochs 100 --test_epoch 5 --evaltype test --save_epochs 1 --seed ${seed} --fix_seed
    # AST
    python train.py --dataset_name AMinerAuthor --embedder whatsnet --att_type_v NoAtt --agg_type_v pure2 --att_type_e NoAtt --agg_type_e pure2 --num_layers 1 --scorer sm --scorer_num_layers 1 --bs 256 --lr 0.001 --sampling 40 --dropout 0.7 --optimizer "adam" --k 0 --gamma 0.99 --dim_hidden 64 --dim_edge 128 --dim_vertex 128 --epochs 100 --test_epoch 5 --evaltype test --save_epochs 1 --seed ${seed} --fix_seed
    # WHATsNET
    python train.py --dataset_name AMinerAuthor --vorder_input "degree_nodecentrality,eigenvec_nodecentrality,pagerank_nodecentrality,kcore_nodecentrality" --embedder whatsnet --att_type_v OrderPE --agg_type_v PrevQ --att_type_e OrderPE --agg_type_e PrevQ --num_att_layer 2 --num_layers 1 --scorer sm --scorer_num_layers 1 --bs 512 --lr 0.001 --sampling 40 --dropout 0.7 --optimizer "adam" --k 0 --gamma 0.99 --dim_hidden 64 --dim_edge 128 --dim_vertex 128 --epochs 100 --test_epoch 5 --evaltype test --save_epochs 1 --seed ${seed} --fix_seed
    # WHATsNET w/o WithinOrderPE
    python train.py --dataset_name AMinerAuthor --embedder whatsnet --att_type_v pure --agg_type_v PrevQ --att_type_e pure --agg_type_e PrevQ --num_att_layer 2 --num_layers 1 --scorer sm --scorer_num_layers 1 --bs 512 --lr 0.001 --sampling 40 --dropout 0.7 --optimizer "adam" --k 0 --gamma 0.99 --dim_hidden 64 --dim_edge 128 --dim_vertex 128 --epochs 100 --test_epoch 5 --evaltype test --save_epochs 1 --seed ${seed} --fix_seed
    # WHATsNET w/o WithinATT
    python train.py --dataset_name AMinerAuthor --embedder whatsnet --att_type_v NoAtt --agg_type_v PrevQ --att_type_e NoAtt --agg_type_e PrevQ --num_layers 1 --scorer sm --scorer_num_layers 1 --bs 512 --lr 0.001 --sampling 40 --dropout 0.7 --optimizer "adam" --k 0 --gamma 0.99 --dim_hidden 64 --dim_edge 128 --dim_vertex 128 --epochs 100 --test_epoch 5 --evaltype test --save_epochs 1 --seed ${seed} --fix_seed
    # WHATsNET-IM
    python train.py --dataset_name AMinerAuthor --vorder_input "degree_nodecentrality,eigenvec_nodecentrality,pagerank_nodecentrality,kcore_nodecentrality" --embedder whatsnet --att_type_v OrderPE --agg_type_v PrevQ --att_type_e OrderPE --agg_type_e PrevQ --num_att_layer 2 --num_layers 1 --scorer im --scorer_num_layers 1 --bs 256 --lr 0.0001 --sampling 40 --dropout 0.7 --optimizer "adam" --k 0 --gamma 0.99 --dim_hidden 64 --dim_edge 128 --dim_vertex 128 --epochs 100 --test_epoch 5 --evaltype test --fix_seed --seed ${seed} --save_epochs 1
done