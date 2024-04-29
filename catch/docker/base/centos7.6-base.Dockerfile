FROM yellow.hub.cambricon.com/pytorch/base/x86_64/pytorch:v0.1-x86_64-centos7.6
#shell

SHELL ["/bin/bash", "-c"]
# change to tsinghua mirror
RUN yum install centos-release-scl-rh epel-release -y
RUN sed -e 's|^mirrorlist=|#mirrorlist=|g' \
        -e 's|^#baseurl=http://mirror.centos.org|baseurl=https://mirrors.tuna.tsinghua.edu.cn|g' \
        -i.bak \
        /etc/yum.repos.d/CentOS-*.repo
RUN yum clean all && yum makecache

ENV LANG C.UTF-8


#install base package
RUN yum install devtoolset-7 -y && \
    scl enable devtoolset-7 bash
ENV PATH=/opt/rh/devtoolset-7/root/usr/bin:$PATH
ENV LD_LIBRARY_PATH=/opt/rh/devtoolset-7/root/usr/lib64:/opt/rh/devtoolset-7/root/usr/lib:/opt/rh/devtoolset-7/root/usr/lib64/dyninst:/opt/rh/devtoolset-7/root/usr/lib/dyninst:/opt/rh/devtoolset-7/root/usr/lib64:/opt/rh/devtoolset-7/root/usr/lib:$LD_LIBRARY_PATH
ENV CC=/opt/rh/devtoolset-7/root/usr/bin/gcc
ENV CXX=/opt/rh/devtoolset-7/root/usr/bin/g++

# libsndfile - used in spectral ops test case.
# numactl - used in MLPerf network for performace, eg: BERT-Large.
RUN yum install -y git make wget zlib-devel bzip2-devel ncurese-devel bzip2 \
    sqlite-devel readline-devel tk-devel boost-devel libSM tkinter patch opencv-devel glog glog-devel openblas-devel \
    tcl-devel gdbm-devel xz-devel libsndfile numactl eigen3-devel perl

# Install IB dependency
RUN yum install -y \
        ca-certificates \
        gnupg \
        wget && \
    rm -rf /var/cache/yum/*
RUN rpm --import https://www.mellanox.com/downloads/ofed/RPM-GPG-KEY-Mellanox && \
    yum install -y yum-utils && \
    yum-config-manager --add-repo https://linux.mellanox.com/public/repo/mlnx_ofed/5.3-1.0.0.1/rhel7.6/mellanox_mlnx_ofed.repo && \
    yum install -y \
        libffi-devel \
        libibumad \
        libibverbs \
        libibverbs-utils \
        librdmacm \
        rdma-core \
        rdma-core-devel && \
    rm -rf /var/cache/yum/*

RUN mkdir -p /tmp/extra-dependencies

# beacause the default openssl version for this system can't meet the needs of python3.10 compiling, so we need install openssl v1.1.1 manually.
RUN cd /tmp && wget https://www.openssl.org/source/openssl-1.1.1.tar.gz && \
    tar -zxvf openssl-1.1.1.tar.gz && cd openssl-1.1.1 && ./config --prefix=/usr/local/openssl shared zlib && \
    make -j && make install && \
    ln -s /usr/local/openssl/bin/openssl /usr/bin/openssl && \
    ln -s /usr/local/openssl/include/openssl /usr/include/openssl && \
    ln -s /usr/local/openssl/lib/libssl.so.1.1 /usr/lib64/libssl.so.1.1 && \
    ln -s /usr/local/openssl/lib/libcrypto.so.1.1 /usr/lib64/libcrypto.so.1.1 && \
    ln -sf /usr/local/openssl/lib/libssl.so.1.1 /usr/lib64/libssl.so && \
    ln -sf /usr/local/openssl/lib/libcrypto.so.1.1 /usr/lib64/libcrypto.so && \
    rm -rf /tmp/openssl-1.1.1.tar.gz && \
    rm -rf /tmp/openssl-1.1.1
# set the openssl certificate to solve verify error on this system.
# https://stackoverflow.com/questions/27835619/urllib-and-ssl-certificate-verify-failed-error
RUN cd /usr/local && wget --quiet https://curl.haxx.se/ca/cacert.pem
ENV SSL_CERT_FILE=/usr/local/cacert.pem

#install latest cmake. See https://askubuntu.com/questions/355565/how-do-i-install-the-latest-version-of-cmake-from-the-command-line
ARG version=3.24
ARG build=1
RUN wget https://cmake.org/files/v$version/cmake-$version.$build.tar.gz && \
    tar -xzvf cmake-$version.$build.tar.gz && \
    pushd cmake-$version.$build/ && \
    ./bootstrap && \
    make -j$(nproc) && \
    make install


# set env
ENV LANG C.UTF-8
ARG python_version="3.10"

# fetch and install python3
# RUN wget -O ~/miniconda.sh https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-4.5.4-Linux-x86_64.sh && \
#     chmod +x ~/miniconda.sh && \
#     ~/miniconda.sh -b -p /opt/conda && \
#     rm ~/miniconda.sh

RUN if [[ ${python_version} == "3.10" ]]; then cd /tmp && \
    wget http://gitlab-software.cambricon.com/neuware/software/framework/pytorch/extra-dependencies/-/raw/master/python/Python-3.10.8.tgz && \
    tar xvf Python-3.10.8.tgz && \
    cd Python-3.10.8 && \
    ./configure --prefix=/opt/py3.10 --enable-ipv6 --with-ensurepip=no  --with-computed-gotos --with-system-ffi --enable-loadable-sqlite-extensions --with-tcltk-includes=-I/opt/py3.10/include '--with-tcltk-libs=-L/opt/py3.10/lib -ltcl8.6 -ltk8.6' --enable-optimizations --with-lto --enable-shared 'CFLAGS=-march=nocona -mtune=haswell -ftree-vectorize -fPIC -fstack-protector-strong -fno-plt -O2 -pipe' 'LDFLAGS=-Wl,-O2 -Wl,--sort-common -Wl,--as-needed -Wl,-z,relro -Wl,-z,now -Wl,-rpath,/opt/py3.10/lib -L/opt/py3.10/lib' 'CPPFLAGS=-DNDEBUG -D_FORTIFY_SOURCE=2 -O2 -I/opt/py3.10/include' PKG_CONFIG_PATH=/opt/py3.10/lib/pkgconfig --with-ensurepip=install && \
    make -j && \
    make altinstall && \
    # use the following 2 lines to compile python without tests
    # make -j8 build_all && \
    # make -j8 altinstall && \
    rm /tmp/Python-3.10.8.tgz && \
    rm -rf /tmp/Python-3.10.8; \
    fi

ENV CPLUS_INCLUDE_PATH=/opt/py${python_version}/include/python${python_version}m:$CPLUS_INCLUDE_PATH
ENV C_INCLUDE_PATH=/opt/py${python_version}/include/python${python_version}m:$C_INCLUDE_PATH

RUN if [[ ${python_version} == "3.10" ]]; then \
    /opt/py3.10/bin/pip3.10 config set global.index-url  http://mirrors.aliyun.com/pypi/simple && \
    /opt/py3.10/bin/pip3.10 config set install.trusted-host mirrors.aliyun.com && \
    /opt/py3.10/bin/pip3.10 install virtualenv && \
    # set ln
    ln -sf /opt/py3.10/bin/python3.10 /usr/bin/python3 && \
    ln -sf /opt/py3.10/bin/pip3.10 /usr/bin/pip && \
    ln -sf /opt/py3.10/bin/virtualenv /usr/bin/virtualenv && \
    ln -sf /usr/bin/cmake3 /usr/bin/cmake; \
    fi

# fetch and install bazel
ARG BAZEL_VERSION="3.4.1"
RUN wget -O ~/bazel.sh http://gitlab.software.cambricon.com/neuware/platform/extra-dependencies/raw/master/bazel/bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh && \
    chmod +x ~/bazel.sh && \
    ~/bazel.sh && \
    rm -f ~/bazel.sh

# Install Open MPI
ENV MPI_HOME=/usr/local/openmpi
ENV PATH=${MPI_HOME}/bin:$PATH
ENV LD_LIBRARY_PATH=${MPI_HOME}/lib:$LD_LIBRARY_PATH

## opencv
RUN mkdir -p /tmp/extra-dependencies
RUN cd /tmp/extra-dependencies && \
    wget https://github.com/opencv/opencv/archive/3.4.14.zip && \
    unzip 3.4.14.zip && \
    cd opencv-3.4.14/ && mkdir build && cd build && \
    cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=/usr/local .. && \
    make -j && make install

## gflags
RUN cd /tmp/extra-dependencies && \
    wget http://gitlab.software.cambricon.com/neuware/software/framework/pytorch/extra-dependencies/-/raw/master/gflags/gflags-2.2.2.zip && \
    unzip gflags-2.2.2.zip && cd gflags && mkdir build && cd build && \
    cmake -DGFLAGS_NAMESPACE=gflags -DBUILD_SHARED_LIBS:BOOL=ON -DCMAKE_CXX_FLAGS="-fPIC -std=c++14 -D _GLIBCXX_USE_CXX11_ABI=0" .. && \
    make -j && make install

RUN rm -rf /tmp/extra-dependencies

RUN mkdir -p $MPI_HOME
RUN cd /tmp && \
    yum -y install bzip2 && \
    wget http://gitlab.software.cambricon.com/neuware/platform/extra-dependencies/-/raw/master/mpi/openmpi-4.1.0.tar.bz2 && \
    tar -jxvf openmpi-4.1.0.tar.bz2 && \
    cd openmpi-4.1.0 && \
    ./configure --prefix=$MPI_HOME --enable-orterun-prefix-by-default && \
    make -j && \
    make install && \
    rm /tmp/openmpi-4.1.0.tar.bz2 && \
    rm -rf /tmp/openmpi-4.1.0
