# Classification of Edge-dependent Labels of Nodes: Formulation, Models, and Benchmark Datasets

We provide (1) benchmark datasets and source code for (2) benchmark task, (3) downstream task and (4) ablation study of WHATsNET

(1) **Benchmark Datasets**

We provide six real-world datasets for our new benchmark task

* Co-authorship : DBLP and AMinerAuthor
* Email : Enron and Eu
* StackOverflow: Biology and Physics

```
# File Organization

|__ hypergraph.txt              # used for constructing hypergraph; i-th line indicates i-th hyperedge includes v_1, v_2, ...
|__ hypergraph_pos.txt          # used for edge-dependent node labels; i-th line indicates depending on i-th hyperedge v_1's label, v_2's label, ... (same order as hypergraph.txt)
|__ [valid/test]_hindex_0.txt   # used for splitting train/valid/test
```
*Due to the large size of the dataset, only 'StackOverflowBiology' and 'DBLP' is visible now in the anonymous GitHub*

(2) **Benchmark Task**

We provide source code for running WHATsNET as well as nine competitors in all the above benchmark datasets

* BaselineU and BaselineP
* HNHN, HGNN, HCHA, HAT, UniGCNII
* HST, AST
* WHATsNET


(3) **Downstream Task**

We apply our benchmark task on the following downstream tasks,

* Ranking Aggregation: https://github.com/uthsavc/hypergraph-halo-ranking
* Clustering: https://github.com/pnnl/HyperNetX/blob/master/tutorials/Tutorial%2011%20-%20Laplacians%20and%20Clustering.ipynb
* Product Return Prediction: https://github.com/jianboli/HyperGo


(4) Reproducing *all* **ablation studies** of WHATsNET

* WithinATT and WithinOrderPE
* Positional Encodings
* WithinATT in propagation from hyperedges to nodes

- - -

## How to Run

### Preprocessing

Before training WHATsNET, calculating node centralities is required

```
cd preprocess
python nodecentrality.py --algo [degree,kcore,pagerank,eigenvec] --dataname [name of dataset]
```

### Run WHATsNET

You can train/evaluate in JSD WHATsNET by
```
python train.py/evaluate.py --vorder_input "degree_nodecentrality,eigenvec_nodecentrality,pagerank_nodecentrality,kcore_nodecentrality" 
                            --embedder transformer --att_type_v OrderPE --agg_type_v PrevQ --att_type_e OrderPE --agg_type_e PrevQ 
                            --dataset_name [name for dataset]
                            --num_att_layer [number of layers in WithinATT]
                            --num_layers [number of layers] 
                            --bs [batch size]
                            --lr [learning rate]
                            --sampling [size of sampling incident hyperedges in aggregation at nodes]
                            --scorer sm --scorer_num_layers 1 --dropout 0.7 --optimizer "adam" --k 0 --gamma 0.99 --dim_hidden 64 --dim_edge 128 --dim_vertex 128 --epochs 100 --test_epoch 5
```

### Run Benchmark Tasks

You can run *all* ten models for each dataset(DBLP,AMinerAuthor,emailEnron,emailEu,StackOverflowBiology,StackOverflowPhyscis) by
```
cd run
./run_[DBLP,AMinerAuthor,emailEnron,emailEu,StackOverflowBiology,StackOverflowPhyscis].sh
```
We set hyperparameters for each model which are chosen by the best mean of Micro-FA and Macro-F1 from the search space

### Run Downstream Tasks

We provide predicted edge-dependent node labels by WHATsNET in `train_results/`

You can run three downstream tasks with WHATsNET by
#### Ranking Aggregation
In the `RankingAggregation` directory, Run `ranking_aggregation_result.ipynb`

#### Clustering
In the `Clustering` directory, Run `clustering.py`

#### Product Return Prediction
In the `ProductReturnPred` directory,

Make synthetic dataset by `makedata/Simulate data.ipynb` and Prepare dataset for training WHATsNET by `makedata/MakeHypergraph.ipynb`

After training WHATsNET,
Run `makedata/Prepare_for_Evaluation_OurModel.ipynb`

Then evaluate on downstreamtask,
```
python main_prod.py               # get result of Hypergraph w/ GroundTruth
python main_prod.py --model_flag  # get result of Hypergraph w/ WHATsNET
python main_prod.py --unif_flag   # get result of Hypergraph w/o Labels
```

### Run Ablation Studies

You can also run *all* ablation studies of WHATsNET by
```
cd run
./run_ablation.sh
```

- - -

## Environment

The environment of running codes is specified in `requirements.txt`
Additionally, install required libraries following `install.sh`