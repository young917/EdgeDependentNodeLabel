dataset=("AMinerAuthor" "emailEu" "StackOverflowBiology")
algos=("degree" "eigenvec" "kcore" "pagerank")

for data in ${dataset[@]}
do
    for algo in ${algos[@]}
    do
        # python nodecentrality.py --dataname DBLP2 --k 0 --exist_hedgename --algo $algo
        python nodecentrality.py --dataname $data --k 0 --algo $algo
    done
done