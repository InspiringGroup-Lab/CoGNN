FROM nvidia/cuda:11.6.1-devel-ubuntu20.04
LABEL maintainer="CoGNN-Authors"
LABEL Description="Image hosting the test environment/dependencies of CoGNN"

WORKDIR /work

ARG DEBIAN_FRONTEND="noninteractive"

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y build-essential git make bzip2 wget && \
    apt-get clean

# Install CMake

RUN apt remove --purge --auto-remove cmake && \
    apt update && \
    apt install -y software-properties-common lsb-release && \
    apt clean all
RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null
RUN apt-add-repository "deb https://apt.kitware.com/ubuntu/ $(lsb_release -cs) main"
RUN apt update && apt install -y cmake

# Install g++/gcc-10

RUN add-apt-repository ppa:ubuntu-toolchain-r/ppa
RUN apt update && apt install -y g++-10 gcc-10
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 10
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-10 20
RUN update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-9 10
RUN update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-10 20
RUN update-alternatives --config gcc && update-alternatives --config g++

RUN apt install -y libgmp-dev libssl-dev
RUN apt install -y build-essential m4 libtool-bin libgmp-dev libntl-dev

# Install Python 3.9

RUN apt install software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt install -y python3.9
RUN ln -sf /usr/bin/python3.9 /usr/bin/python 
RUN apt install -y python3-pip
RUN python -m pip install psutil pandas

RUN apt-get install -y net-tools iproute2
RUN apt-get install -y iptables
RUN apt-get install -y vim

ADD ./Container-Artifact/work /work/

# Set up EMP and usr/local/cmake

RUN cd /work/MPC/emp-tool && make install && \
    cd /work/MPC/emp-ot && make install && \
    cd /work/MPC/emp-sh2pc && make install && \
    cd /work/MPC && mkdir /usr/local/lib/cmake/ && cp -r cmake/* /usr/local/lib/cmake/

# Set up troy

RUN cd /work/MPC/troy/build && rm -r * && \
    cmake .. -DCMAKE_INSTALL_PREFIX=./install && cmake --build . --target install --parallel

# Set up CoGNN

RUN cd /work/Art/build && rm -r * && \
    cmake .. -DTHREADING=ON && make -j8
