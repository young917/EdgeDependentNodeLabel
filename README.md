# Classification of Edge-dependent Labels of Nodes in Hypergraphs

We provide (1) datasets and source code for (2) benchmark task, (3) downstream task and (4) ablation study of WHATsNET

(1) **Datasets**

We provide six real-world datasets for our new benchmark task(```/dataset/```) and preprocessing code (```/dataset/PreprocessCode/```)

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
* HNHN, HGNN, HCHA, HAT, UniGCNII, HNN
* HST, AST
* WHATsNET


(3) **Downstream Task**

We apply our benchmark task on the following downstream tasks,

* Ranking Aggregation: https://github.com/uthsavc/hypergraph-halo-ranking
* Clustering: https://github.com/pnnl/HyperNetX/blob/master/tutorials/Tutorial%2011%20-%20Laplacians%20and%20Clustering.ipynb
* Product Return Prediction: https://github.com/jianboli/HyperGo


(4) Reproducing *ALL* results in Paper

* **Ablation Studies** of WHATsNET
* w/o WithinATT and WithinOrderPE
* WHATsNET-IM
* Positional encodings schemes
* Replacing WithinATT in updating node embeddings
* Number of inducing points
* Types of node centralities
* **Visualization** of WHATsNET
* **Evaluation on Node Label Distribution Preservation** of WHATsNET

- - -

## How to Run

### Preprocessing

Before training WHATsNET, calculating node centralities is required

```
cd preprocess
python nodecentrality.py --algo [degree,kcore,pagerank,eigenvec] --dataname [name of dataset]
```

### Run WHATsNET

You can 

* train WHATsNET
* evaluate WHATsNET on JSD of node-level label dist.
* predict edge-dependent node labels by trained WHATsNET
* analysis node embeddings for visualization: concatenated embeddings of a node and hyperedge pair, node embeddings before/after WithinATT 

by following below code,
```
python train.py/evaluate.py/predict/analysis.py  --vorder_input "degree_nodecentrality,eigenvec_nodecentrality,pagerank_nodecentrality,kcore_nodecentrality" 
                                                 --embedder whatsnet --att_type_v OrderPE --agg_type_v PrevQ --att_type_e OrderPE --agg_type_e PrevQ 
                                                 --dataset_name [name for dataset]
                                                 --num_att_layer [number of layers in WithinATT]
                                                 --num_layers [number of layers] 
                                                 --bs [batch size]
                                                 --lr [learning rate]
                                                 --sampling [size of sampling incident hyperedges in aggregation at nodes]
                                                 [--analyze_att  when running analysis.py]
                                                 --scorer sm --scorer_num_layers 1 --dropout 0.7 --optimizer "adam" --k 0 --gamma 0.99 --dim_hidden 64 --dim_edge 128 --dim_vertex 128 --epochs 100 --test_epoch 5
```

### Run Benchmark Tasks

You can run *all* ten models for each dataset(DBLP,AMinerAuthor,emailEnron,emailEu,StackOverflowBiology,StackOverflowPhyscis) by
```
cd run
./run_[DBLP,AMinerAuthor,emailEnron,emailEu,StackOverflowBiology,StackOverflowPhyscis].sh
```
We set hyperparameters of each model chosen by the best mean of Micro-F1 and Macro-F1 from the search space

### Run Downstream Tasks

We provide edge-dependent node labels predicted by WHATsNET as well as AST and HST in `train_results/`
We also provide shell script for all-in-one (train, predict and evaluate on the downstream task) in `run/DownstreamTask/`

You can run three downstream tasks with WHATsNET and baselines by
#### Ranking Aggregation
In the `RankingAggregation` directory, 

For Halo2 game dataset, run `ranking_aggregation_result.ipynb` or `ranking.py`

For AMiner dataset with author H-index, run `aminer_ranking.py`

#### Clustering
In the `Clustering` directory, 

For DBLP, run `clustering.py`

For AMiner, run `clustering_aminer.py`

#### Product Return Prediction
In the `ProductReturnPred` directory,

Make synthetic dataset by `makedata/Simulate data.ipynb` and Prepare dataset for training models by our benchmark task through `makedata/MakeHypergraph.ipynb`

After training models, run `makedata/prepare_predicted.py`

Then evaluation of the product return prediction can be done by `script/main_prod.py`

### Run Ablation Studies

You can also run *all* ablation studies of WHATsNET by
```
cd run
./run_ablation.sh
./run_ablation_centrality.sh
```

- - -

## Environment

The environment of running codes is specified in `requirements.txt`
Additionally, install required libraries following `install.sh`
