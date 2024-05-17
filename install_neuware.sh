#!/usr/bin/env bash

set -e

mkdir neuware_pkg
cd neuware_pkg

# CNToolkit
wget https://sdk.cambricon.com/static/Basis/MLU370_X86_ubuntu20.04/cntoolkit_3.7.2-1.ubuntu20.04_amd64.deb
sudo dpkg -i cntoolkit_3.7.2-1.ubuntu20.04_amd64.deb && sudo apt update && sudo apt install cntoolkit-cloud
sudo rm -f /etc/apt/sources.list.d/cntoolkit.list

# CNNL
wget https://sdk.cambricon.com/static/Basis/MLU370_X86_ubuntu20.04/cnnl_1.21.1-1.ubuntu20.04_amd64.deb
sudo apt install ./cnnl_1.21.1-1.ubuntu20.04_amd64.deb

# CNCL
wget https://sdk.cambricon.com/static/Basis/MLU370_X86_ubuntu20.04/cncl_1.13.0-1.ubuntu20.04_amd64.deb
sudo apt install ./cncl_1.13.0-1.ubuntu20.04_amd64.deb

# CNCV
wget https://sdk.cambricon.com/static/Basis/MLU370_X86_ubuntu20.04/cncv_2.3.0-1.ubuntu20.04_amd64.deb
sudo apt install ./cncv_2.3.0-1.ubuntu20.04_amd64.deb

# CNNL_Extra
wget https://sdk.cambricon.com/static/Basis/MLU370_X86_ubuntu20.04/cnnlextra_1.5.0-1.ubuntu20.04_amd64.deb
sudo apt install ./cnnlextra_1.5.0-1.ubuntu20.04_amd64.deb

# MLU-OPS
wget https://sdk.cambricon.com/static/Basis/MLU370_X86_ubuntu20.04/mluops_0.9.0-1.ubuntu20.04_amd64.deb
sudo apt install ./mluops_0.9.0-1.ubuntu20.04_amd64.deb

# Cambricon DALI
wget https://sdk.cambricon.com/static/Basis/MLU370_X86_ubuntu20.04/cambricon_dali-0.9.0-py3-none-manylinux2014_x86_64.whl
pip install cambricon_dali-0.9.0-py3-none-manylinux2014_x86_64.whl

# Set environment variables
echo 'export NEUWARE_HOME=/usr/local/neuware' >> ~/.bashrc
echo 'export PATH=$NEUWARE_HOME/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$NEUWARE_HOME/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc

echo "Cambricon Neuware Package Installation successful!"
echo "NEUWARE_HOME: /usr/local/neuware"