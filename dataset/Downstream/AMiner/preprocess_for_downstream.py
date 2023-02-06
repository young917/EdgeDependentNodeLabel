from collections import defaultdict
import numpy as np

OUTPUTDIR = "/workspace/WithInHyperedgeNodeLabel/RankingAggregation/"

reference = defaultdict(list)
citation = defaultdict(list)
pub2paperid = defaultdict(list)
paperid2pub = {}
paperid2title = {}
paperid2year = {}

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
                    paperid2pub[paperid] = content.replace(",", " ")
                elif sp == "#t":# year
                    if len(content) == 0:
                        content = "-1"
                    paperid2year[paperid] = int(content)
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

with open(OUTPUTDIR + "paper_info.txt", "w") as f:
    f.write("paperid,venue,year\n")
    for paperid in paperid2year.keys():
        f.write(str(paperid) + "," + str(paperid2pub[paperid]) + "," + str(paperid2year[paperid]) + "\n")

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
                    authorid2weight[authorid]["t"] = content
                elif sp == "#n":
                    authorid2weight[authorid]["name"] = content
                line = f.readline().rstrip()
                if len(line) == 0:
                    break
            author_count += 1
        line = f.readline().rstrip()

with open(OUTPUTDIR + "author_info.txt", "w") as f:
    f.write("authorID,PaperCount,Citation,HIndex,PIndex,AIndex,Interest\n")
    for authorid in authorid2weight.keys():
        line = [str(authorid)]
        for e in ["pc","cn","hi","pi","upi","t"]:
            line.append(str(authorid2weight[authorid][e]))
        f.write(",".join(line) + "\n")

hedge2paperid = {}
paperid2hedge = {}
authorid2node = {}
node2authorid = {}

hedge2node = []
hedge2nodepos = []
node2hedge = []
numhedges = 0
numnodes = 0

# index, authorid, paperid, author_pos
with open("AMiner-Author2Paper.txt", "r") as f:
    for line in f.readlines():
        _, authorid, paperid, pos = line.rstrip().split("\t")
        authorid, paperid, pos = int(authorid), int(paperid), int(pos)
        
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
        # hedge2node[hidx].append(vidx)
        hedge2node[hidx].append(authorid)
        hedge2nodepos[hidx].append(pos)
        node2hedge[vidx].append(hidx)
        
print("Done AMiner-Author2Paper")

check_hedges = [0 for _ in range(numhedges)]
# hypergraph, hypergraph_pos
with open(OUTPUTDIR + "hypergraph.txt", "w") as f, open(OUTPUTDIR + "hypergraph_pos.txt", "w") as pf:
    for hidx in range(numhedges):
        nodes = hedge2node[hidx]
        paperid = hedge2paperid[hidx]
        nodenames = []
        # for v in nodes:
            # authorid = node2authorid[v]
        for authorid in nodes:
            nodename = authorid2weight[authorid]["name"]
            nodenames.append(nodename)
        if np.array_equal(np.arange(len(nodes)), np.argsort(nodenames)):
            continue
        check_hedges[hidx] = 1
        
        line = [str(paperid)] + [str(v) for v in hedge2node[hidx]]
        line = "\t".join(line)
        f.write(line + "\n")

        line = [str(paperid)] + [str(vpos) for vpos in hedge2nodepos[hidx]]
        line = "\t".join(line)
        pf.write(line + "\n")
        
with open(OUTPUTDIR + "hypergraph_rank.txt", "w") as f:
    for vidx in range(numnodes):
        authorid = node2authorid[vidx]
        f.write(str(authorid) + "\t" + str(authorid2weight[authorid]["hi"]) + "\n")
        
with open(OUTPUTDIR + "hypergraph_citation.txt", "w") as f:
    for hidx in range(numhedges):
        if check_hedges[hidx] == 1:
            paperid = hedge2paperid[hidx]
            numcitation = len(citation[paperid])
            f.write(str(paperid) + "\t" + str(numcitation) + "\n")
