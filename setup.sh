#!/bin/sh

#git clone https://github.com/ksangeeta2429/l3embedding.git
#cd l3embedding/
git checkout dcompression
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
bash ~/miniconda.sh -b -p ~/miniconda
rm ~/miniconda.sh
echo "PATH=$PATH:$HOME/miniconda/bin" >> .bashrc
source .bashrc
conda create -q -n l3embedding-new-cpu python=3.6.2
source activate l3embedding-new-cpu
pip install -r requirements_cpu.txt
pip install --upgrade google-api-python-client google-auth-httplib2 google-auth-oauthlib
python gsheets.py l3compression ~/credentials.json --noauth_local_webserver
source deactivate
