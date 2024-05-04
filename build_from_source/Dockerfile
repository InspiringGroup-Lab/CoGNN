FROM cbackyx/cognn-ae-build:v1
LABEL maintainer="CoGNN-Authors"
LABEL Description="Image hosting the test environment/dependencies of CoGNN, building from sources."

WORKDIR /work

ARG DEBIAN_FRONTEND="noninteractive"

# Set up EMP and usr/local/cmake

RUN cd /work/MPC/emp-tool && cmake . -DTHREADING=ON && make -j && make install && \
    cd /work/MPC/emp-ot && cmake . -DTHREADING=ON && make -j && make install && \
    cd /work/MPC/emp-sh2pc && cmake . -DTHREADING=ON && make -j && make install && \
    cd /work/MPC && mkdir -p /usr/local/lib/cmake/ && cp -r cmake/* /usr/local/lib/cmake/

# Set up troy

RUN cd /work/MPC/troy/build && \
    cmake .. -DCMAKE_INSTALL_PREFIX=./install && cmake --build . --target install --parallel

# Set up ophelib

RUN cd /work/MPC/ophelib/build && \
    cmake .. -DCMAKE_INSTALL_PREFIX=./install && cmake --build . --target install --parallel

# Set up libOTe

RUN cd /work/Art/libOTe && \
    python build.py --setup --boost --relic && \
    python build.py --install=./../lilibOTe -- -D ENABLE_RELIC=ON -D ENABLE_NP=ON -D ENABLE_KOS=ON -D ENABLE_IKNP=ON -D ENABLE_OOS=ON -D ENABLE_SILENTOT=ON

# Set up SCI-SilentOT

RUN cd /work/MPC/SCI-SilentOT && \
    bash scripts/build-deps.sh && \
    cd build && \
    cmake -DCMAKE_INSTALL_PREFIX=./install -DCMAKE_FIND_DEBUG_MODE=ON .. -DCMAKE_BUILD_TYPE=Debug -DSCI_BUILD_NETWORKS=OFF -DSCI_BUILD_TESTS=ON -DCMAKE_PREFIX_PATH=$(pwd)/../deps/build -DUSE_APPROX_RESHARE=ON
    
RUN cd /work/MPC/SCI-SilentOT/build && cmake --build . --target install --parallel

# Set up CoGNN

RUN cd /work/Art/build && \
    cmake .. -DTHREADING=ON && make -j
