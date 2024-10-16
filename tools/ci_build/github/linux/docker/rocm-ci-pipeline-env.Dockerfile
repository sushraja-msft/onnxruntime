# Refer to https://github.com/RadeonOpenCompute/ROCm-docker/blob/master/dev/Dockerfile-ubuntu-22.04-complete
FROM ubuntu:22.04

ARG ROCM_VERSION=6.2
ARG AMDGPU_VERSION=${ROCM_VERSION}
ARG APT_PREF='Package: *\nPin: release o=repo.radeon.com\nPin-Priority: 600'

# The local directory into which to build and install CMAKE
ARG ONNXRUNTIME_LOCAL_CODE_DIR=/code

CMD ["/bin/bash"]

RUN echo "$APT_PREF" > /etc/apt/preferences.d/rocm-pin-600

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates curl libnuma-dev gnupg && \
    curl -sL https://repo.radeon.com/rocm/rocm.gpg.key | apt-key add -   &&\
    printf "deb [arch=amd64] https://repo.radeon.com/rocm/apt/$ROCM_VERSION/ jammy main" | tee /etc/apt/sources.list.d/rocm.list   && \
    printf "deb [arch=amd64] https://repo.radeon.com/amdgpu/$AMDGPU_VERSION/ubuntu jammy main" | tee /etc/apt/sources.list.d/amdgpu.list   && \
    apt-get update && apt-get install -y --no-install-recommends \
    sudo \
    libelf1 \
    kmod \
    file \
    libstdc++6 \
    python3 \
    python3-pip \
    python3.10-dev \
    python3.10-venv \
    rocm-dev \
    rocm-libs \
    build-essential && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN groupadd -g 109 render

# Upgrade to meet security requirements
RUN apt-get update -y && apt-get upgrade -y && apt-get autoremove -y && \
    apt-get install  -y locales cifs-utils wget half libnuma-dev lsb-release && \
    apt-get clean -y

RUN locale-gen en_US.UTF-8
RUN update-locale LANG=en_US.UTF-8
ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8

WORKDIR /stage

# Cmake
ENV CMAKE_VERSION=3.30.1
RUN cd /usr/local && \
    wget -q https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-Linux-x86_64.tar.gz && \
    tar -zxf /usr/local/cmake-3.30.1-Linux-x86_64.tar.gz --strip=1 -C /usr

# ccache
RUN mkdir -p /tmp/ccache && \
    cd /tmp/ccache && \
    wget -q -O - https://github.com/ccache/ccache/releases/download/v4.7.4/ccache-4.7.4-linux-x86_64.tar.xz | tar --strip 1 -J -xf - && \
    cp /tmp/ccache/ccache /usr/bin && \
    rm -rf /tmp/ccache

WORKDIR /ort

RUN python3 -m venv /ort/env \
    && . /ort/env/bin/activate \
    && pip install packaging \
                   setuptools \
                   numpy==2.1.2 \
                   wheel \
                   onnx==1.16.1 \
                   protobuf==4.21.12 \
                   sympy==1.12 \
                   flatbuffers \
                   psutil \
                   ml_dtypes==0.3.0 \
                   pytest \
                   pytest-xdist \
                   pytest-rerunfailures \
                   scipy==1.10.0

RUN apt install -y git

# Install Cupy to decrease CPU utilization
RUN git clone https://github.com/ROCm/cupy && cd cupy && \
    git checkout 432a8683351d681e00903640489cb2f4055d2e09 && \
    export CUPY_INSTALL_USE_HIP=1 && \
    export ROCM_HOME=/opt/rocm && \
    export HCC_AMDGPU_TARGET=gfx906,gfx908,gfx90a && \
    git submodule update --init && \
    pip install -e . --no-cache-dir -vvvv
