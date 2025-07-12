FROM ubuntu:20.04
# hmmm

LABEL maintainer="JDM jm4622@ic.ac.uk"

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai
ENV LC_ALL=C.UTF-8

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    tzdata \
    wget curl unzip git vim sudo \
    build-essential \
    ninja-build \
    cmake \
    clang lld lldb clang-format \
    gdb \
    python3 python3-dev python3-pip python3-venv \
    virtualenv \
    swig \
    libomp-dev \
    libgl1 \
    libnuma1 libatlas-base-dev \
    libncurses5-dev libncurses5 \
    graphviz \
    openssh-server openssh-client \
    rsync \
    bsdmainutils \
    ca-certificates \
    && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime \
    && echo $TZ > /etc/timezone \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# install libssl1.1 for sophgo
RUN wget http://nz2.archive.ubuntu.com/ubuntu/pool/main/o/openssl/libssl1.1_1.1.1f-1ubuntu2.24_amd64.deb && \
    dpkg -i libssl1.1_1.1.1f-1ubuntu2.24_amd64.deb && \
    rm libssl1.1_1.1.1f-1ubuntu2.24_amd64.deb

# install eiq
COPY eiq-toolkit-v1.12.1.77-1_amd64_b240708.deb.bin.zip /tmp/

RUN cd /tmp && \
    unzip eiq-toolkit-v1.12.1.77-1_amd64_b240708.deb.bin.zip && \
    chmod +x eiq-toolkit-v1.12.1.77-1_amd64_b240708.deb.bin && \
    ./eiq-toolkit-v1.12.1.77-1_amd64_b240708.deb.bin && \
    # optionally remove installer files to keep image clean
    rm -f eiq-toolkit-v1.12.1.77-1_amd64_b240708.deb.bin eiq-toolkit-v1.12.1.77-1_amd64_b240708.deb.bin.zip

RUN python3 -m pip install --upgrade pip==22.0.2 setuptools==59.6.0 wheel==0.37.1

# install CMake 3.25.3
RUN wget https://github.com/Kitware/CMake/releases/download/v3.25.3/cmake-3.25.3-linux-x86_64.sh -O /tmp/cmake-install.sh \
    && chmod +x /tmp/cmake-install.sh \
    && /tmp/cmake-install.sh --skip-license --prefix=/usr/local \
    && rm /tmp/cmake-install.sh

WORKDIR /workspace

COPY requirements.txt /workspace/requirements.txt
RUN python3 -m pip install --upgrade pip setuptools wheel && python3 -m pip install --no-cache-dir -r /workspace/requirements.txt

COPY src/ /workspace/src/

RUN git config --global --add safe.directory '*'

# RUN cd tpu-mlir
# RUN source ./envsetup.sh
# RUN ./build.sh
# RUN cd ..

CMD ["/bin/bash"]
