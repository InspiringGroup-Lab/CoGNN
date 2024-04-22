CoGNN Artifact Evaluation Guidance
============

> Evaluating these artifacts requires a Linux Server (at least 128GB RAM, 512GB spare disk) equipped with NVIDIA GPU (at least 16GB Memory). Please make sure of this before reading ahead.

This document is dedicated to the Artifact Evaluation Committee (AEC) of our paper *CoGNN: Towards Secure and Efficient Collaborative Graph Learning*. It provides some necessary background information about the paper and contains step-by-step instructions for setting up the artifacts and running the vital experiments in our paper. Here is the table of contents:
- *0 Background*: some brief background information about CoGNN;
- *1 Introduction*: introduce these artifacts and their organization;
- *2 Environment Requirements*: the required hardware resources and software environments;
- *3 Set Up*: set up the environment by downloading the Docker image we provide;
- *4 Evaluation*: steps to run each part of the experiments.  

## 0 Background

CoGNN is a provably secure, efficient and scalable framework for collaborative graph learning. In collaborative graph learning, there are multiple parties (graph owners), each owning a private local graph. The local graphs of different parties are interleaved by some inter-edges, and they finally consititute a global graph. For example, in financial scenarios, each party can be a bank, then each local graph is the transfer graph inside a graph, and inter-edges are inter-bank transfers. These local transfer graphs are concatenated into a global transfer graph by inter-bank transfers. The ultimate goal of CoGNN is to have the parties jointly train a Graph Neural Network (GNN) model on the global graph.

Performing GNN learning on the global graph yields a more comprehensive view than learning on a local graph. However, the precondition is protecting the local graph data privacy of each party. We want the parties to jointly compute (train or inference) a global model, but do not want to leak their local graph to any other parties. Overall, CoGNN leverages cryptographic techniques, including secret share, secure multi-party computation (MPC) and homomorphic encryption (HE) to achieve this privacy-preserving property.    

The core features of CoGNN are two-fold: 

- CoGNN works in an in-place computation fashion where the graph owners act as the computing parties. It never outsources the raw gragh data of each participant, in order to honor the raw data privacy. It is built upon 2-party secure computation backends, but can support involving an arbitrary number of parties. This is different from prior works that are also built upon MPC, which outsouce the graph data to a fixed number of third-party computing parties.
- CoGNN fully embraces distributed computation among all the parties and requires no centralized (trusted) server. The distributed computation makes it scalable as more parties are involved and the size of the global graph grows. 

In the paper, we compare CoGNN to two branches of related works:
- Efficiency (running duration & per-party communication) comparison with Secure Machine Learning (SML)-based works. These works achieve the same level of privacy guarantee and model accuracy as CoGNN. So we compare with them for efficiency.
- Accuracy comparison with Federated learning (FL)-based works. These works are faster than CoGNN since they use plaintext computation, but are thus also privacy-leaking. Additionally, they can not handle inter-edges, limiting the accuracy of the global model. So we compare model accuracy to them, in order to show that inter-edges are important for model accuracy.

## 1 Introduction

These artifacts correspond to our prototype implementation of the CoGNN and our evaluations of it. Currently, we implement the training and inference of Graph Convolutional Network (GCN). 

Here is the organization of the artifacts in the Docker image we provide:

```bash 
└── 📁work # The overall workspace
    └── 📁Art # The main part of our artifacts
        └── 📁CoGNN
            └── 📁algo_kernels
                └── 📁common_harness # The entrance of the built executable
                └── 📁vertex_centric
                    └── 📁optimize-gcn # Defines the Scatter-Gather-Apply (GAS) operations for our optimized version of GCN training
                    └── 📁optimize-gcn-inference # Defines the Scatter-Gather-Apply (GAS) operations for our optimized version of GCN inference
                    └── 📁original-gcn # Defines the Scatter-Gather-Apply (GAS) operations for our unoptimized version of GCN training
            └── 📁include
                └── ...
                └── ss_vertex_centric_algo_kernel.h # The CoGNN engine, defining per-iteration computation flow. The `onIteration' function is the client function, leading two-party tasks with private information (the orders of vertex and edge lists). The `runAlgoKernelServer' function is the server function, coordinating with a leading client for two-party tasks.
                └── 📁task
                └── 📁utils
            └── makefile.inc
            └── 📁tools
                └── 📁plot # Some plot functions.
                └── 📁scripts # Scripts for setting up simulated network environments, using network namespace.
                └── tmp_run_cluster.py # The scripts for running the experiments.
        └── 📁Task-Worker
            └── 📁include
                └── ObliviousMapper.h # The definition of OEP 
                └── SCIHarness.h # The wrapper for the SCI-SilentOT backend for GNN 2PC computations
                └── SecureAggregation.h # The definition of OGA
            └── 📁src
                └── ObliviousMapper.cpp
                └── SCIHarness.cpp
                └── SecureAggregation.cpp
            └── 📁test
                └── 📁GraphAnalysis # The implementation of the SML-based state-of-the-art
                └── 2PC_test.cpp # The unit tests for 2PC computations
                └── fed_gcn.cpp # The FedAvg-based federated learning approach
                └── graphsc_test.cpp # The SML-based state-of-the-art
                └── plaintext_gcn.cpp # Plaintext global training
    └── 📁MPC # The dependencies
        └── 📁SCI-SilentOT # For generic secure two-party computation
        └── 📁cmake
        └── install.py # For install EMP-related dependencies
        └── 📁ophelib # For Paillier (This is not used in current implementation)
        └── 📁troy # For GPU-accelerated FHE
```

## 2 Environment Requirements

The source code of CoGNN is available in the repositories owned by this anonymous Github account. However, since building each part of the artifacts along with the dependencies is non-trivial, we provide an out-of-the-box Docker image for convenience.

We summarize the required hardware resources and software conditions for running the Docker image we provide.

**Hardware Resources**
- A x86-64 Linux server (at least 128GB RAM, 512GB spare disk)
    - We tested on Intel(R) Xeon(R) Gold 6348 CPU @ 2.60GHz
- NVIDIA GPU (at least 16GB Memory)
    - We tested on NVIDIA A100 80GB PCIe and NVIDIA GeForce RTX 4090

**Software Resources**
- Operating System
    - We tested on Ubuntu 20.04 (with APT package manager)
- CUDA Driver and Toolkit
    - We tested on (Driver version 535.113.01, CUDA version 12.2) and (Driver version 550.54.14, CUDA version 12.4)
- Docker with CUDA support (nvidia-container-toolkit)
    - We tested on Docker version 24.0.5

Note that, you need to install and **configure** nvidia-container-toolkit for your Docker, as following:
```bash
# Install
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
# Configure
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker # Do not forget this step!
```
Please refer to [link 1](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html), [link 2](https://stackoverflow.com/questions/25185405/using-gpu-from-a-docker-container).

## 3 Set Up



## 0 Set up the dependencies

### 0.1 Environment Requirement

- Ubuntu 20.04
- g++/gcc 9.4.0
- GNU Make 4.2.1
- cmake 3.25.1
- Python 3.9.13
- CUDA Driver Version: 535.113.01  CUDA Version: 12.2

### 0.2 Set up the workspace

```bash
mkdir Artifact && cd Artifact
mkdir Art
```

### 0.3 Install MPC dependencies.

Install EMP.

```bash
cd Artifact
git clone https://github.com/CoGNN-anon/MPC.git
cd MPC
python install.py --deps --tool --ot --sh2pc
```

Install SCI-SilentOT.

```bash
sudo apt install libgmp-dev libssl-dev
cd Artifact/MPC/
git clone https://github.com/CoGNN-anon/SCI-SilentOT.git
cd ./SCI-SilentOT/SCI
bash scripts/build-deps.sh
mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX=./install -DCMAKE_FIND_DEBUG_MODE=ON .. -DCMAKE_BUILD_TYPE=Debug -DSCI_BUILD_NETWORKS=OFF -DSCI_BUILD_TESTS=ON -DOPENSSL_ROOT_DIR=/usr/local/opt/openssl -DCMAKE_PREFIX_PATH=$(pwd)/../deps/build -DUSE_APPROX_RESHARE=ON
cmake --build . --target install --parallel
```

Install ophelib (Paillier).

```bash
sudo apt install build-essential m4 libtool-bin libgmp-dev libntl-dev
cd Artifact/MPC
git clone https://github.com/CoGNN-anon/ophelib.git
cd ophelib
mkdir build && cd build
cmake ..
make -j
sudo make install
```

Install troy (GPU-based FHE).

```bash
cd Artifact/MPC
git clone https://github.com/CoGNN-anon/troy.git
cd troy
mkdir build && cd build
cmake ..
cmake --build . --target install --parallel
```

### 0.4 Set Up CMake Dependencies

```bash
cd Artifact/MPC
sudo cp -r cmake/* /usr/local/lib/cmake/
```

### 0.5 Set Up OT

```bash
cd Artifact/Art/
git clone https://github.com/osu-crypto/libOTe.git
cd libOTe
python build.py --install=./../lilibOTe -- -D ENABLE_RELIC=ON -D ENABLE_NP=ON -D ENABLE_KOS=ON -D ENABLE_IKNP=ON -D ENABLE_OOS=ON -D ENABLE_SILENTOT=ON
```

## 1 Set up CoGNN

> Note that, you might have to manually set some paths in the CMakeList.txt according to your environment.

```bash
cd Artifact/Art/

git clone https://github.com/CoGNN-anon/CoGNN.git
git clone https://github.com/CoGNN-anon/Task-Worker.git

mkdir build && cd build
cmake .. -DTHREADING=ON
make -j8
```

## 2 Prepare the datasets

```bash
cd Artifact/Art/CoGNN/tools
python data_transform.py
```

## 3 Run evaluation

```bash
cd Artifact/Art/CoGNN/tools
python tmp_run_cluster.py
```


