#pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
#pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.11.0+cu113.html
pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.13.0+cu116.html
# pip install dgl-cu113 dglgo -f https://data.dgl.ai/wheels/repo.html
pip install dgl-cu116 dglgo -f https://data.dgl.ai/wheels/repo.html
pip install gensim
pip install hypernetx
pip install trueskill