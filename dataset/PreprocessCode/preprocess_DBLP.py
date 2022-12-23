from collections import defaultdict

def isInAlphabeticalOrder(word):
    for i in range(len(word) - 1):
        if word[i] > word[i + 1]:
            return False
    return True

hypergraph = defaultdict(list) # papercode: [(nodeindex, pos) ... ]
authors2info = {} # nodeindex: name, gender, genderprob, affiliation, lat, lng
hedges2info = defaultdict(dict) # year, conf, cs, de, se, th
authorkey2index = {} # name -> nodeindex
# Hist
affiliation_hist = defaultdict(int)
conference_hist = defaultdict(int)

with open("nodeinfo.txt", "w") as of:
    of.write("nodeindex\tname\tgender\tgenderprob\taffiliation\tlat\tlng\n")

with open("hyperedgeinfo.txt", "w") as of:
    of.write("papercode\tyear\tconf\tcs\tde\tse\tth\n")

# Fill paper <-> author, author's name, gender, gender_prob
with open("authors.txt", "r", encoding="utf-8") as f:
    c = f.read(1)
    assert c == "("

    onedata = ""
    nodenum = 0
    while True:
        c = f.read(1)
        if not c:
            break
        onedata += c
        if len(onedata) >=3 and onedata[-3:] == "),(":
            # process
            line = onedata[:-3]
            _, papercode, pos, name, gender, gender_prob = line.split(",")
            if name not in authorkey2index:
                nodeindex = nodenum
                authorkey2index[name] = nodeindex
                nodenum += 1
                authors2info[nodeindex] = {"name": name, "gender": gender, "gender_prob": gender_prob}
            nodeindex = authorkey2index[name]
            hypergraph[papercode].append((nodeindex, pos))
            onedata = ""
    
    assert onedata[-1] == ")", onedata
    # last element
    line = onedata[:-1]
    _, papercode, pos, name, gender, gender_prob = line.split(",")
    if name not in authorkey2index:
        nodeindex = nodenum
        authorkey2index[name] = nodeindex
        nodenum += 1
        authors2info[nodeindex] = {"name": name, "gender": gender, "gender_prob": gender_prob}
    nodeindex = authorkey2index[name]
    hypergraph[papercode].append((nodeindex, pos))

# Fill remain information: affiliation, lat, lng
with open("affiliation_coord.txt", "r", encoding="utf-8") as f:
    c = f.read(1)
    assert c == "("

    onedata = ""
    nodes = set()
    count = 0
    while True:
        c = f.read(1)
        if not c:
            break
        elif c == "'" and (len(onedata) > 0 and onedata[-1] != "\\"):
            count += 1
        elif c == "," and (count % 2 == 1):
            c = "-"
        onedata += c
        if len(onedata) >=3 and onedata[-3:] == "),(":
            # process
            line = onedata[:-3]
            if len(line.split(",")) != 9:
                print(line)
            _, papercode, name, affiliation, year, contry2, country_code, lat, lng = line.split(",")
            if name not in nodes:
                nodes.add(name)
                assert name in authorkey2index, name
                nodeindex = authorkey2index[name]
                authors2info[nodeindex]["affiliation"] = affiliation
                authors2info[nodeindex]["lat"] = lat
                authors2info[nodeindex]["lng"] = lng
                affiliation_hist[affiliation] += 1
            onedata = ""
            count = 0
    
    assert onedata[-1] == ")", onedata
    # last element
    line = onedata[:-1]
    _, papercode, name, affiliation, year, contry2, country_code, lat, lng = line.split(",")
    if name not in nodes:
        nodes.add(name)
        assert name in authorkey2index
        nodeindex = authorkey2index[name]
        authors2info[nodeindex]["affiliation"] = affiliation
        authors2info[nodeindex]["lat"] = lat
        authors2info[nodeindex]["lng"] = lng
        affiliation_hist[affiliation] += 1

# Fill hyperedge(paper)'s information
with open("main.txt", "r") as f:
    c = f.read(1)
    assert c == "("

    onedata = ""
    while True:
        c = f.read(1)
        if not c:
            break
        onedata += c
        if len(onedata) >=3 and onedata[-3:] == "),(":
            # process
            line = onedata[:-3]
            if len(line.split(",")) != 10:
                print(line)
            papercode, year, conf, papercode2, cs, de, se, th, publisher, link = line.split(",")
            if papercode in hypergraph: # right?            
                hedges2info[papercode] = {"year": year, "conference": conf, "cs": cs, "de": de, "se": se, "th": th}
                conference_hist[conf] += 1
            onedata = ""
    
    assert onedata[-1] == ")", onedata
    # last element
    line = onedata[:-1]
    papercode, year, conf, papercode2, cs, de, se, th, publisher, link = line.split(",")
    assert papercode in hypergraph
    hedges2info[papercode] = {"year": year, "conference": conf, "cs": cs, "de": de, "se": se, "th": th}
    conference_hist[conf] += 1

'''
authors2info = {} # nodeindex, gender, genderprob, affiliation,lat,lng
hedges2info = defaultdict(dict) # year, conf, cs, de, se, th
authorkey2index = {}
# Hist
affiliation_hist = defaultdict(int)
conference_hist = defaultdict(int)
'''

# Write output ----------------------------------------------------------------------------------------------

# authors2info = {} # nodeindex: name, gender, genderprob, affiliation, lat, lng
check_hyperedge = {}
for h in hypergraph.keys():
    check_hyperedge[h] = True
filtered = 0
with open("hypergraph.txt", "w") as of, open("hypergraph_pos.txt", "w") as pf:
    for h in hypergraph:
        # preprocess alphabetic order
        temp = []
        for n in hypergraph[h]:
            nodeindex, pos = n
            nodename = authors2info[nodeindex]['name']
            temp.append((int(pos), nodename))
        temp = sorted(temp, key=lambda x: x[0])
        namelist = [x[1] for x in temp]
        if isInAlphabeticalOrder(namelist):
            check_hyperedge[h] = False
            filtered += 1
            continue
        
        of.write(h)
        pf.write(h)
        for n in hypergraph[h]:
            nodeindex, pos = n
            of.write("\t" + str(nodeindex))
            pf.write("\t" + pos)
        of.write("\n")
        pf.write("\n")     
print(filtered, "is filtered out of", len(check_hyperedge))

with open("nodeinfo.txt", "a", encoding='utf8') as nf:
    for nodeindex, val in authors2info.items():
        for c in ["affiliation", "lat", "lng"]:
            if c not in val:
                val[c] = "NULL"
        tmp = [str(nodeindex), val['name'], val['gender'], val['gender_prob'], val['affiliation'], val['lat'], val['lng']]
        line = '\t'.join(tmp) + "\n"
        nf.write(line)

with open("hyperedgeinfo.txt", "a") as ef:
    for c in ["year", "conf", "cs", "de", "se", "th"]:
            if c not in val:
                val[c] = "NULL"
    for papercode, val in hedges2info.items():
        if check_hyperedge[papercode]:
            ef.write(papercode + "\t" + val["year"] + "\t" + val["conference"]+ "\t" + val["cs"] + "\t" + val["de"] + "\t" + val["se"] + "\t" + val["th"] + "\n")

with open("affiliation_hist.txt", "w", encoding='utf8') as f:
    for aff in affiliation_hist:
        f.write(aff + "\t" + str(affiliation_hist[aff]) + "\n")
        
with open("conference_hist.txt", "w") as f:
    for conf in conference_hist:
        f.write(conf + "\t" + str(conference_hist[conf]) + "\n")

