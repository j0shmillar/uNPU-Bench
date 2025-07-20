FROM sophgo/tpuc_dev:latest

LABEL maintainer="JDM jm4622@ic.ac.uk"

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai
ENV LC_ALL=C.UTF-8

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    tzdata \
    wget curl unzip git vim sudo \
    build-essential \
    ninja-build \
    cmake \
    clang lld lldb clang-format \
    gdb \
    python3.10 python3.10-dev python3.10-venv python3.10-distutils \
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

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10 && \
    update-alternatives --install /usr/bin/pip3 pip3 /usr/local/bin/pip3 1

# install libssl1.1 for sophgo
RUN wget http://nz2.archive.ubuntu.com/ubuntu/pool/main/o/openssl/libssl1.1_1.1.1f-1ubuntu2.24_amd64.deb && \
    dpkg -i libssl1.1_1.1.1f-1ubuntu2.24_amd64.deb && \
    rm libssl1.1_1.1.1f-1ubuntu2.24_amd64.deb

# install eiq
COPY eiq-toolkit-v1.12.1.77-1_amd64_b240708.deb.bin.zip /tmp/

RUN cd /tmp && \
    unzip eiq-toolkit-v1.12.1.77-1_amd64_b240708.deb.bin.zip && \
    chmod +x eiq-toolkit-v1.12.1.77-1_amd64_b240708.deb.bin && \
    ./eiq-toolkit-v1.12.1.77-1_amd64_b240708.deb.bin --noexec --target ./eiq_extract && \
    dpkg -i ./eiq_extract/*.deb && \
    rm -rf /tmp/eiq-toolkit* /tmp/eiq_extract

RUN python3 -m pip install --upgrade pip==22.0.2 setuptools==59.6.0 wheel==0.37.1

RUN wget https://github.com/Kitware/CMake/releases/download/v3.25.3/cmake-3.25.3-linux-x86_64.sh -O /tmp/cmake-install.sh \
    && chmod +x /tmp/cmake-install.sh \
    && /tmp/cmake-install.sh --skip-license --prefix=/usr/local \
    && rm /tmp/cmake-install.sh

WORKDIR /workspace

COPY requirements.txt /workspace/requirements.txt
RUN python3 -m pip install --upgrade pip setuptools wheel && \
    python3 -m pip install --no-cache-dir -r /workspace/requirements.txt

COPY src/ /workspace/src/

RUN git config --global --add safe.directory '*'

RUN cd src && git clone --recursive https://github.com/analogdevicesinc/ai8x-training.git && \
    git clone --recursive https://github.com/analogdevicesinc/ai8x-synthesis.git

ENV AI8X_TRAIN_PATH=/workspace/src/ai8x-training/

CMD ["/bin/bash"]
