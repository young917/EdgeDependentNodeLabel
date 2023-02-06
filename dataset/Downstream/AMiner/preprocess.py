from collections import defaultdict

reference = defaultdict(list)
citation = defaultdict(list)
pub2paperid = defaultdict(list)
paperid2pub = {}
paperid2title = {}

# paper id - (#*) paper title
# paper id - (#%) -> count -> citation count
# paper id - (#c) publication venue
paper_count = 0
with open("AMiner-Paper.txt", "r") as f:
    line = f.readline().rstrip()
    while True:
        if not line:
            break
        if paper_count % 10000 == 0:
            print("{} done ...".format(paper_count))
        if line.startswith("#index"):
            paperid = int(line.split(" ")[-1])
            line = f.readline().rstrip()
            while line.startswith("#index") is False:
                tmp = line.split(" ")
                sp = tmp[0]
                content = " ".join(tmp[1:])
                if sp == "#*": # paper title
                    paperid2title[paperid] = content
                elif sp == "#c": # publication venue
                    pub2paperid[content].append(paperid)
                    paperid2pub[paperid] = content
                elif sp == "#%":
                    refid = int(content)
                    reference[paperid].append(refid)
                    citation[refid].append(paperid)
                line = f.readline().rstrip()
                if len(line) == 0:
                    break
            paper_count += 1
        
        line = f.readline().rstrip()

print("Done AMiner-Paper")
# print(pub2paperid.keys())
with open("publist.txt", "w") as f:
    for pub in pub2paperid.keys():
        f.write(pub + "\n")

target = []
target = defaultdict(list)
indexing = {"NIPS" : 5, "ICML": 1, "KDD": 2, "IJCAI": 3, "UAI": 4} # COLT
pub2index = {}
for pub in pub2paperid.keys():
    if "Advances in neural information processing systems" in pub:
        target["NIPS"].append(pub)
        pub2index[pub] = indexing["NIPS"]
    elif "ICML " in pub:
        target["ICML"].append(pub)
        pub2index[pub] = indexing["ICML"]
    elif "kdd" in pub.lower():
        _pub = pub.lower()
        if "acm sigkdd international conference" in _pub:
            target["KDD"].append(pub)
            pub2index[pub] = indexing["KDD"]
        else:
            exist_flag = False
            for rem in ["pkdd", "pakdd", "padkk", "webkdd", "tkdd", "wkdd", "pinkdd", "snakdd", "sensor-kdd", "hci-kdd", "tutorial", "newsletter", "workshop", "kdd cup", "kdd loop"]:
                if rem in _pub:
                    exist_flag = True
                    break
            if exist_flag is False:
                target["KDD"].append(pub)
                pub2index[pub] = indexing["KDD"]
    elif "ijcai" in pub.lower():
        if "workshop" in pub.lower():
            continue
        if "joint conference on artificial intelligence" in pub.lower():
            target["IJCAI"].append(pub)
            pub2index[pub] = indexing["IJCAI"]
        elif "jont conference on artifical intelligence" in pub.lower():
            target["IJCAI"].append(pub)
            pub2index[pub] = indexing["IJCAI"]
        elif "joint conference on artifical intelligence" in pub.lower():
            target["IJCAI"].append(pub)
            pub2index[pub] = indexing["IJCAI"]
    elif "UAI" in pub:
        target["UAI"].append(pub)
        pub2index[pub] = indexing["UAI"]
    

list(pub2paperid.keys())
insampled_paper = defaultdict(int)
for reppub in target:
    print(reppub)
    print(target[reppub])
    for pub in target[reppub]:
        for paperid in pub2paperid[pub]:
            insampled_paper[paperid] = 1
    print()


#index ---- index id of this author
#n ---- name  (separated by semicolons)
#a ---- affiliations  (separated by semicolons)
#pc ---- the count of published papers of this author
#cn ---- the total number of citations of this author
#hi ---- the H-index of this author
#pi ---- the P-index with equal A-index of this author
#upi ---- the P-index with unequal A-index of this author
#t ---- research interests of this author  (separated by semicolons)
authorid2weight = defaultdict(dict)
interest = defaultdict(int)
author_count = 0
with open("AMiner-Author.txt", "r") as f:
    line = f.readline().rstrip()
    while True:
        if not line:
            break
        if author_count % 10000 == 0:
            print("{} done ...".format(author_count))
        if line.startswith("#index"):
            authorid = int(line.split(" ")[-1])
            line = f.readline().rstrip()
            while line.startswith("#index") is False:
                tmp = line.split(" ")
                sp = tmp[0]
                content = " ".join(tmp[1:])
                if sp == "#pc": # count of published papers
                    authorid2weight[authorid]["pc"] = int(content)
                elif sp == "#cn": # total number of citations
                    authorid2weight[authorid]["cn"] = int(content)
                elif sp == "#hi": # h-index
                    authorid2weight[authorid]["hi"] = float(content)
                elif sp == "#pi": # p-index
                    authorid2weight[authorid]["pi"] = float(content)
                elif sp == "#upi": # a-index
                    authorid2weight[authorid]["upi"] = float(content)
                elif sp == "#t":
                    for it in content.split(";"):
                        interest[it] += 1
                line = f.readline().rstrip()
                if len(line) == 0:
                    break
            author_count += 1
        
        line = f.readline().rstrip()


hedge2paperid = {}
paperid2hedge = {}
hedge2node = []
hedge2nodepos = []
authorid2node = {}
node2authorid = {}
node2hedge = []
numhedges = 0
numnodes = 0
# index, authorid, paperid, author_pos
with open("AMiner-Author2Paper.txt", "r") as f:
    for line in f.readlines():
        _, authorid, paperid, pos = line.rstrip().split("\t")
        authorid, paperid, pos = int(authorid), int(paperid), int(pos)
        if insampled_paper[paperid] == 0:
            continue
        if paperid not in paperid2hedge:
            hidx = numhedges
            paperid2hedge[paperid] = hidx
            hedge2paperid[hidx] = paperid
            numhedges += 1
            hedge2node.append([])
            hedge2nodepos.append([])
        if authorid not in authorid2node:
            vidx = numnodes
            authorid2node[authorid] = vidx
            node2authorid[vidx] = authorid
            numnodes += 1
            node2hedge.append([])
        hidx = paperid2hedge[paperid]
        vidx = authorid2node[authorid]
        hedge2node[hidx].append(vidx)
        hedge2nodepos[hidx].append(pos)
        node2hedge[vidx].append(hidx)
print("Done AMiner-Author2Paper")

# hypergraph, hypergraph_pos
with open("hypergraph.txt", "w") as f, open("hypergraph_pos.txt", "w") as pf:
    for hidx in range(numhedges):
        line = [str(v) for v in hedge2node[hidx]]
        line = "\t".join(line)
        f.write(line + "\n")

        line = [str(vpos) for vpos in hedge2nodepos[hidx]]
        line = "\t".join(line)
        pf.write(line + "\n")
    
# hyperedge_cluster
with open("hyperedge_cluster.txt", "w") as f:
    for hidx in range(numhedges):
        paperid = hedge2paperid[hidx]
        pub = paperid2pub[paperid]
        pubidx = pub2index[pub]
        f.write(str(pubidx) + "\n")

# hyperedge_weight
with open("hyperedge_weight.txt", "w") as f:
    for hidx in range(numhedges):
        paperid = hedge2paperid[hidx]
        refs = [id for id in reference[paperid] if id in paperid2hedge]
        ref_count = len(refs)
        cits = [id for id in citation[paperid] if id in paperid2hedge]
        cit_count = len(cits)

        f.write(str(ref_count) + "\t" + str(cit_count) + "\n")

keys = ["pc", "cn", "hi", "pi", "upi"]
with open("node_weight.txt", "w") as f:
    for vidx in range(numnodes):
        authorid = node2authorid[vidx]
        line = [str(vidx)]
        for k in keys:
            if k not in authorid2weight[authorid]:
                line.append("0")
            else:
                line.append(str(authorid2weight[authorid][k]))
        f.write("\t".join(line) + "\n")
        