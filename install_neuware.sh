#!/usr/bin/env bash

set -e

mkdir -p neuware_pkg
cd neuware_pkg

# check os version
version=$(lsb_release -rs)
if [[ "$version" != "18.04" && "$version" != "20.04" ]]; then
    echo "Error：Unsupported version '$version'. Only supports Ubuntu 18.04 and Ubuntu 20.04"
    exit 1
fi

# CNToolkit
if ! dpkg -s cntoolkit >/dev/null 2>&1; then
    wget -nc https://sdk.cambricon.com/static/Basis/MLU370_X86_ubuntu${version}/cntoolkit_3.7.2-1.ubuntu${version}_amd64.deb
    sudo dpkg -i cntoolkit_3.7.2-1.ubuntu${version}_amd64.deb
    sudo apt update
    sudo apt install -y cntoolkit-cloud
    sudo rm -f /etc/apt/sources.list.d/cntoolkit.list
else
    echo "Info: CNToolkit is already installed."
fi

# CNNL
if ! dpkg -s cnnl >/dev/null 2>&1; then
    wget -nc https://sdk.cambricon.com/static/Basis/MLU370_X86_ubuntu${version}/cnnl_1.21.1-1.ubuntu${version}_amd64.deb
    sudo apt install -y ./cnnl_1.21.1-1.ubuntu${version}_amd64.deb
else
    echo "Info: CNNL is already installed."
fi

# CNCL
if ! dpkg -s cncl >/dev/null 2>&1; then
    wget -nc https://sdk.cambricon.com/static/Basis/MLU370_X86_ubuntu${version}/cncl_1.13.0-1.ubuntu${version}_amd64.deb
    sudo apt install -y ./cncl_1.13.0-1.ubuntu${version}_amd64.deb
else
    echo "Info: CNCL is already installed."
fi

# CNCV
if ! dpkg -s cncv >/dev/null 2>&1; then
    wget -nc https://sdk.cambricon.com/static/Basis/MLU370_X86_ubuntu${version}/cncv_2.3.0-1.ubuntu${version}_amd64.deb
    sudo apt install -y ./cncv_2.3.0-1.ubuntu${version}_amd64.deb
else
    echo "Info: CNCV is already installed."
fi

# CNNL_Extra
if ! dpkg -s cnnlextra >/dev/null 2>&1; then
    wget -nc https://sdk.cambricon.com/static/Basis/MLU370_X86_ubuntu${version}/cnnlextra_1.5.0-1.ubuntu${version}_amd64.deb
    sudo apt install -y ./cnnlextra_1.5.0-1.ubuntu${version}_amd64.deb
else
    echo "Info: CNNL_Extra is already installed."
fi

# MLU-OPS
if ! dpkg -s mluops >/dev/null 2>&1; then
    wget -nc https://sdk.cambricon.com/static/Basis/MLU370_X86_ubuntu${version}/mluops_0.9.0-1.ubuntu${version}_amd64.deb
    sudo apt install -y ./mluops_0.9.0-1.ubuntu${version}_amd64.deb
else
    echo "Info: MLU-OPS is already installed."
fi

# Cambricon DALI
if ! pip list | grep -F cambricon-dali >/dev/null 2>&1; then
    wget -nc https://sdk.cambricon.com/static/Basis/MLU370_X86_ubuntu${version}/cambricon_dali-0.9.0-py3-none-manylinux2014_x86_64.whl
    pip install cambricon_dali-0.9.0-py3-none-manylinux2014_x86_64.whl
else
    echo "Info: Cambricon DALI is already installed."
fi


echo ""
echo "==================================================================="
echo "Cambricon Neuware Package Installation successful!"
echo "NEUWARE_HOME: /usr/local/neuware"

# Set environment variables
lines_to_add=(
    "export NEUWARE_HOME=/usr/local/neuware"
    "export PATH=\$NEUWARE_HOME/bin:\$PATH"
    "export LD_LIBRARY_PATH=\$NEUWARE_HOME/lib64:\$LD_LIBRARY_PATH"
)
bashrc="$HOME/.bashrc"
for line in "${lines_to_add[@]}"; do
    if ! grep -Fxq "$line" $bashrc; then
        echo "$line" >> $bashrc
        echo "Added '$line' to $bashrc"
    fi
done