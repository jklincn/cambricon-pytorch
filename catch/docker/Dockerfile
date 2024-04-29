ARG py_suffix="py310"
FROM yellow.hub.cambricon.com/pytorch/base/x86_64/pytorch:v0.9-x86_64-manylinux2014-torch_py310_gcc7_mpi

MAINTAINER CATCH

ARG catch_branch="catch_1.13.1_develop"
ARG vision_branch="vision_0.14.1"
ARG pytorch_branch="pytorch_1.13.1"
ARG python_version="3.10"

# neuware lib version
ARG driver_version
ARG cntoolkit_version
ARG cnnl_version
ARG cnnlextra_version
ARG cncl_version
ARG magicmind_version

LABEL "com.cambricon.driver"=${driver_version}       \
      "com.cambricon.cntoolkit"=${cntoolkit_version} \
      "com.cambricon.cnnl"=${cnnl_version}           \
      "com.cambricon.cnnlextra"=${cnnlextra_version} \
      "com.cambricon.cncl"=${cncl_version}           \
      "com.cambricon.magicmind"=${magicmind_version}
RUN if [[ ${python_version} == "3.10" ]]; then mkdir /torch /wheel_py310; fi
# fetch pytorch codes
RUN pushd /torch && \
    echo $pytorch_branch && \
    git clone http://gitlab.software.cambricon.com/neuware/pytorch.git -b $pytorch_branch --depth 1 && \
    popd

# fetch catch codes
RUN pushd /torch && \
    echo $catch_branch && \
    git clone http://gitlab.software.cambricon.com/neuware/catch.git -b $catch_branch --depth 1 && \
    popd

# fetch vision codes
RUN pushd /torch && \
    echo $vision_branch && \
    git clone http://gitlab.software.cambricon.com/neuware/vision.git -b $vision_branch --depth 1 && \
    popd

ENV PYTORCH_HOME=/torch/pytorch
RUN pushd /torch/catch/scripts&& ./apply_patches_to_pytorch.sh && popd

RUN pushd /torch && bash /torch/catch/scripts/release/independent_build.sh -r dep -o centos -v 7

ENV NEUWARE_HOME=/torch/neuware_home
ENV LD_LIBRARY_PATH=/torch/neuware_home/lib64:$LD_LIBRARY_PATH
ENV USE_NINJA=OFF
ENV CC=/opt/rh/devtoolset-7/root/usr/bin/gcc
ENV CXX=/opt/rh/devtoolset-7/root/usr/bin/g++
ENV TORCH_CXX_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=0 -fvisibility=hidden -Wl,--exclude-libs,ALL"
ENV CXXFLAGS="-D_GLIBCXX_USE_CXX11_ABI=0"
ENV GLIBCXX_USE_CXX11_ABI=0

# set pip source
RUN PIP_CONF_DIR=${HOME}/.config/pip && mkdir -p ${PIP_CONF_DIR} && \
    echo "[global]" > ${PIP_CONF_DIR}/pip.conf && \
    echo "index-url = https://pypi.tuna.tsinghua.edu.cn/simple/" \
          >> ${PIP_CONF_DIR}/pip.conf && \
    echo "extra-index-url = http://mirrors.cambricon.com/pypi/web/simple/" >> ${PIP_CONF_DIR}/pip.conf && \
    echo "[install]" >> ${PIP_CONF_DIR}/pip.conf && \
    echo "trusted-host = pypi.tuna.tsinghua.edu.cn" >> ${PIP_CONF_DIR}/pip.conf && \
    echo "               mirrors.cambricon.com" >> ${PIP_CONF_DIR}/pip.conf

# build torch-py3.10

ENV CPLUS_INCLUDE_PATH=/opt/py${python_version}/include/python${python_version}m:$CPLUS_INCLUDE_PATH
ENV C_INCLUDE_PATH=/opt/py${python_version}/include/python${python_version}m:$C_INCLUDE_PATH
RUN if [[ ${python_version} == "3.10" ]]; then pushd /torch && \
    /opt/py3.10/bin/virtualenv -p /opt/py3.10/bin/python3.10 venv3/pytorch && \
    source venv3/pytorch/bin/activate && \
    pushd /torch/pytorch && \
    pip install -r requirements.txt && \
    pushd /torch/pytorch && \
    pip install setuptools==59.5.0 && \
    python setup.py bdist_wheel && \
    pip install dist/*.whl && \
    pushd /torch/catch && \
    pip install -r requirements.txt && \
    pushd /torch/catch && \
    python setup.py bdist_wheel && \
    pushd /torch/vision && \
    python setup.py bdist_wheel && \
    mv /torch/pytorch/dist/*.whl /wheel_py310 && \
    mv /torch/catch/dist/*.whl /wheel_py310 && \
    mv /torch/vision/dist/*.whl /wheel_py310; \
    fi
