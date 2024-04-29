FROM yellow.hub.cambricon.com/base/centos:8.3-x86_64

#shell
SHELL ["/bin/bash", "-c"]

# change to tsinghua mirror
RUN minorver=8.4.2105 && sed -e "s|^mirrorlist=|#mirrorlist=|g" \
        -e "s|^#baseurl=http://mirror.centos.org/\$contentdir/\$releasever|baseurl=https://mirrors.tuna.tsinghua.edu.cn/centos-vault/$minorver|g" \
        -i.bak \
        /etc/yum.repos.d/CentOS-*.repo

RUN yum clean all && yum makecache
RUN yum install centos-release epel-release -y

#install base package
# RUN yum install devtoolset-7 -y && \
#     scl enable devtoolset-7 bash
RUN dnf group install "Development Tools" -y
# ENV PATH=/usr/bin:$PATH
# ENV LD_LIBRARY_PATH=/opt/rh/devtoolset-7/root/usr/lib64:/opt/rh/devtoolset-7/root/usr/lib:/opt/rh/devtoolset-7/root/usr/lib64/dyninst:/opt/rh/devtoolset-7/root/usr/lib/dyninst:/opt/rh/devtoolset-7/root/usr/lib64:/opt/rh/devtoolset-7/root/usr/lib:$LD_LIBRARY_PATH
ENV CC=/usr/bin/gcc
ENV CXX=/usr/bin/g++
# ENV PATH=/opt/rh/devtoolset-7/root/usr/bin:$PATH
# ENV LD_LIBRARY_PATH=/opt/rh/devtoolset-7/root/usr/lib64:/opt/rh/devtoolset-7/root/usr/lib:/opt/rh/devtoolset-7/root/usr/lib64/dyninst:/opt/rh/devtoolset-7/root/usr/lib/dyninst:/opt/rh/devtoolset-7/root/usr/lib64:/opt/rh/devtoolset-7/root/usr/lib:$LD_LIBRARY_PATH
# ENV CC=/opt/rh/devtoolset-7/root/usr/bin/gcc
# ENV CXX=/opt/rh/devtoolset-7/root/usr/bin/g++

# RUN yum install -y git make cmake cmake3 wget zlib-devel bzip2-devel openssl-devel ncurese-devel bzip2 gflags-devel \
#     sqlite-devel readline-devel tk-devel boost-devel libSM tkinter patch opencv opencv-devel glog glog-devel openblas-devel \
#     tcl-devel gdbm-devel xz-devel libsndfile

# libsndfile - used in spectral ops test case.
# numactl - used in MLPerf network for performace, eg: BERT-Large.
RUN yum install -y git make cmake cmake3 wget zlib-devel bzip2-devel openssl-devel bzip2 sqlite-devel readline-devel tk-devel boost-devel libSM python3-tkinter patch opencv-core tcl-devel gdbm-devel xz-devel libsndfile numactl perl
RUN dnf --enablerepo=powertools install -y glog glog-devel openblas-devel eigen3-devel
# see https://blog.csdn.net/chenyulancn/article/details/118540210 and http://jira.cambricon.com/browse/PYTORCH-7512
RUN yum install -y libarchive
# Install IB dependency
RUN yum install -y \
        ca-certificates \
        gnupg \
        wget && \
    rm -rf /var/cache/yum/*
RUN rpm --import https://www.mellanox.com/downloads/ofed/RPM-GPG-KEY-Mellanox && \
    yum install -y dnf-utils && \
    yum-config-manager --add-repo https://linux.mellanox.com/public/repo/mlnx_ofed/5.2-2.2.0.0/rhel8.0/mellanox_mlnx_ofed.repo && \
    yum install -y \
        libffi-devel \
        libibumad \
        libibverbs \
        libibverbs-utils \
        librdmacm \
        rdma-core \
        rdma-core-devel && \
    rm -rf /var/cache/yum/*

# set env
ENV LANG C.UTF-8
ARG python_version="3.10"

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
    ln -sf /opt/py3.10/bin/virtualenv /usr/bin/virtualenv; \
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
