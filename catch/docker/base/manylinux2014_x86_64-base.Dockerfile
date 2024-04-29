FROM yellow.hub.cambricon.com/pytorch/base/x86_64/pytorch:v0.1-x86_64-manylinux2014

# change to tsinghua mirror
RUN sed -e 's|^mirrorlist=|#mirrorlist=|g' \
        -e 's|^#baseurl=http://mirror.centos.org|baseurl=https://mirrors.tuna.tsinghua.edu.cn|g' \
        -i.bak \
        /etc/yum.repos.d/CentOS-*.repo
RUN yum clean all && yum makecache

# install gcc-7
RUN rm -rf /opt/rh/devtoolset-9 && \
    yum install devtoolset-7 -y && \
    scl enable devtoolset-7 bash
RUN echo "source /opt/rh/devtoolset-7/enable" >> ~/.bashrc && \
    echo "source /opt/rh/devtoolset-7/enable" >> /etc/profile

ENV PATH=/opt/rh/devtoolset-7/root/bin:$PATH
ENV LD_LIBRARY_PATH=/opt/rh/devtoolset-7/root/usr/lib64:/opt/rh/devtoolset-7/root/usr/lib:/opt/rh/devtoolset-7/root/usr/lib64/dyninst:/opt/rh/devtoolset-7/root/usr/lib/dyninst:/opt/rh/devtoolset-7/root/usr/lib64:/opt/rh/devtoolset-7/root/usr/lib:$LD_LIBRARY_PATH
ENV CC=/opt/rh/devtoolset-7/root/usr/bin/gcc
ENV CXX=/opt/rh/devtoolset-7/root/usr/bin/g++

# install some dependence 
RUN yum install -y glog-devel openblas-devel rsync wget boost-devel tcl-devel tk-devel bzip2-devel ncurses-devel gdbm-devel xz-devel sqlite-devel readline-devel tkinter 

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

# beacause the default openssl version for this system can't meet the needs of python3.10 compiling, so we need install openssl v1.1.1 manually.
RUN cd /tmp && wget --no-check-certificate https://www.openssl.org/source/openssl-1.1.1.tar.gz && \
    tar -zxvf openssl-1.1.1.tar.gz && cd openssl-1.1.1 && ./config --prefix=/usr/local/openssl shared zlib && \
    make -j && make install && \
    mv /usr/bin/openssl /usr/bin/openssl_back && \
    ln -s /usr/local/openssl/bin/openssl /usr/bin/openssl && \
    ln -s /usr/local/openssl/include/openssl /usr/include/openssl && \
    ln -s /usr/local/openssl/lib/libssl.so.1.1 /usr/lib64/libssl.so.1.1 && \
    ln -s /usr/local/openssl/lib/libcrypto.so.1.1 /usr/lib64/libcrypto.so.1.1 && \
    ln -sf /usr/local/openssl/lib/libssl.so.1.1 /usr/lib64/libssl.so && \
    ln -sf /usr/local/openssl/lib/libcrypto.so.1.1 /usr/lib64/libcrypto.so && \
    rm -rf /tmp/openssl-1.1.1.tar.gz && \
    rm -rf /tmp/openssl-1.1.1

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
