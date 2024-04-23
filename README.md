CoGNN Artifact Evaluation Guidance
============

> *Cautions*: 
> - Evaluating these artifacts requires an x86_64 Linux Server (at least 128GB RAM, 512GB spare disk) equipped with NVIDIA GPU (at least 16GB Memory).
> - We pack our artifacts in a Docker image. The compressed size is around 4~6GB. We provide instructions for setting up CUDA support for your Docker.  
> - Running a smallest test of our artifacts needs around 1 minute, but fully running all the experiments in our paper might take 3 days or more.
> - Please make sure of these details before deciding to read ahead.

This document is dedicated to the Artifact Evaluation Committee (AEC) of our paper *CoGNN: Towards Secure and Efficient Collaborative Graph Learning*. It provides some necessary background information about the paper and contains step-by-step instructions for setting up the artifacts and running the vital experiments in our paper. Here is the table of contents:
- [0 Background](#0-background): some brief background information about CoGNN;
- [1 Introduction](#1-introduction): introduce these artifacts and their organization;
- [2 Environment Requirements](#2-environment-requirements): the required hardware resources and software environments;
- [3 Set Up](#3-set-up): set up the artifacts by downloading the Docker image we provide;
- [4 Evaluation](#4-evaluation): steps to run each part of the experiments.  


## 0 Background

CoGNN is a provably secure, efficient and scalable framework for collaborative graph learning. In collaborative graph learning, there are multiple parties (graph owners), each owning a private local graph. The local graphs of different parties are interleaved by some inter-edges, and they finally consititute a global graph. For example, in financial scenarios, each party can be a bank, then each local graph is the transfer graph inside a graph, and inter-edges are inter-bank transfers. These local transfer graphs are concatenated into a global transfer graph by inter-bank transfers. The ultimate goal of CoGNN is to have the parties jointly train a Graph Neural Network (GNN) model on the global graph.

Performing GNN learning on the global graph yields a more comprehensive view than learning on a local graph. However, the precondition is protecting the local graph data privacy of each party. We want the parties to jointly compute (train or inference) a global model, but do not want to leak their local graph to any other parties. Overall, CoGNN leverages cryptographic techniques, including secret share, secure multi-party computation (MPC) and homomorphic encryption (HE) to achieve this privacy-preserving property.    

The core features of CoGNN are two-fold: 

- CoGNN works in an in-place computation fashion where the graph owners act as the computing parties. It never outsources the raw gragh data of each participant, in order to honor the raw data privacy. It is built upon 2-party secure computation backends, but can support involving an arbitrary number of parties. This is different from prior works that are also built upon MPC, which outsouce the graph data to a fixed number of third-party computing parties.
- CoGNN fully embraces distributed computation among all the parties and requires no centralized (trusted) server. The distributed computation makes it scalable as more parties are involved and as the size of the global graph grows. 

In the paper, we compare CoGNN to two branches of related works:
- *Efficiency & scalability (running duration & per-party communication) comparison* with Secure Machine Learning (SML)-based works. These works achieve the same level of privacy guarantee and model accuracy as CoGNN. So we compare with them for efficiency.
- *Accuracy (model performance) comparison* with Federated learning (FL)-based works. These works are faster than CoGNN since they use plaintext computation, but are thus also privacy-leaking. Additionally, they can not handle inter-edges, limiting the accuracy of the global model. So we compare model accuracy to them, in order to show that inter-edges are important for model accuracy.

## 1 Introduction

These artifacts correspond to our prototype implementation of CoGNN and our evaluations of it. Currently, we implement the training and inference of Graph Convolutional Network (GCN). 

Our evaluations include:
- *Efficiency & Scalability*: measure the *running duration* and *communication* of each party during training (1 epoch) and inference (1 full-graph inference)
    - Compared schemes: CoGNN, CoGNN-Opt (optimized CoGNN), GraphSC (SML-based SOTA)
- *Accuracy*: measure model accuracies on *the whole test dataset* and *the border test dataset*, after training for 90 epochs
    - Compared schemes: CoGNN-Opt, FedGNN (FL-based, FedAvg), PlaintextGNN (plaintext global graph training)

Here is the organization of the artifacts in the Docker image we provide:

```bash 
└── 📁work # The overall workspace
    └── 📁Art # The main part of our artifacts
        └── 📁CoGNN
            └── 📁algo_kernels
                └── 📁common_harness # The main function (executable entrance)
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
                └── 📁data # The data prepared for our evaluations
                    └── 📁CiteSeer
                    └── 📁PubMed                
                    └── 📁Cora
                        └── 📁processed
                        └── 📁raw
                        └── 📁transformed # From 2s to 5s are for efficiency evaluation. The others are for accuracy evaluation
                            └── 📁2s
                                └── cora.edge.preprocessed # Edge list
                                └── cora.part.preprocessed # Partition info (each vertex belongs to which party)
                                └── cora.vertex.preprocessed # Vertex list
                                └── cora_config.txt # Training hyperparameters
                            └── 📁3s
                            └── 📁4s
                            └── 📁5s
                            └── cora.edge.preprocessed
                            └── cora.part.preprocessed.2p
                            └── cora.part.preprocessed.3p
                            └── cora.part.preprocessed.4p
                            └── cora.part.preprocessed.5p
                            └── cora.vertex.preprocessed
                            └── cora_config.txt # Training hyperparameters
                └── 📁plot # Some plot functions.
                └── 📁scripts # Scripts for setting up simulated network environments, using network namespace.
                └── tmp_run_cluster.py # The scripts for running the experiments.
        └── 📁Task-Worker # MPC & HE-based computations for CoGNN
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
        └── 📁ophelib # For Paillier (This is not used in the current implementation)
        └── 📁troy # For GPU-accelerated FHE
```

## 2 Environment Requirements

The source code of CoGNN is available in the repositories owned by [this anonymous Github account](https://github.com/CoGNN-anon). However, since building each part of the artifacts along with the dependencies is non-trivial, we provide an out-of-the-box Docker image for convenience.

We summarize the required hardware resources and software conditions for running the Docker image we provide.

**Hardware Resources**
- An x86_64 Linux server (at least 128GB RAM, 512GB spare disk)
    - We tested on Intel(R) Xeon(R) Gold 6348 CPU @ 2.60GHz with 512GB RAM
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

As long as you have installed Docker with CUDA support, you can pull the image we prepared. Our image is based on the Ubuntu-CUDA images provided by NIVIDIA. Since these images are bound to specific versions of CUDA runtime, we provide multiple image versions based on different versions of CUDA runtimes for the convenience of the AEC. The version information is summarized here:
- v1 is based on nvidia/cuda:12.0.0-devel-ubuntu20.04
- v2 is based on nvidia/cuda:12.4.0-devel-ubuntu20.04
- v3 is based on nvidia/cuda:12.3.2-devel-ubuntu20.04
- v4 is based on nvidia/cuda:11.6.1-devel-ubuntu20.04

Please pull the proper version according to CUDA driver version and CUDA version you have. (The compressed image size is around 4~7GB. The duration (~10min or more) depends on your network condition.)
Typically the CUDA runtime version has to be smaller than the CUDA toolkit version of your host machine.

```bash
sudo docker pull cbackyx/cognn-ae:<tagname>
```

*Caution:* In case that none of the image versions above satisfy your need, we also provide a workaround by partially building the image on your own using Dockerfile.
The idea is to copy the artifacts in our image to the image that you are going to build. (This building process would take longer time, also depending on your network condition.)
Ignore this if we have already provided your desired version.

```bash
# Pull one of the images above first and run.
sudo docker pull cbackyx/cognn-ae:v1
sudo docker run -it --rm --privileged --security-opt apparmor=unconfined --gpus all cbackyx/cognn-ae:v1 /bin/bash
# Now you can see your <container-id>. Copy it.
# Open another terminal on your host machine (not in the container!).
# cd to a proper clean workspace and copy the Dockerfile here. Then:
mkdir Container-Artifact
# Copy the artifacts from the container you just started.
sudo docker cp <container-id>:/work ./Container-Artifact/
# Modify the first line of the Dockerfile to select the CUDA runtime version you want. (You MUST use a devel version, since we need CUDA headers for compilation.)
# Please refer to https://hub.docker.com/r/nvidia/cuda/tags?page=1&page_size=&name=devel-ubuntu20.04&ordering=
# Now you are ready to build the image.
sudo docker build -t cbackyx/cognn-ae:tagname .
```

Now let's have a quick check on the artifacts.
Build the executables that we need (~5s):
```bash
# Enter the container first. Please select the proper tagname for yourself.
sudo docker run -it --rm --privileged --security-opt apparmor=unconfined --gpus all cbackyx/cognn-ae:<tagname> /bin/bash
# Check if your GPU can be accessed
nvidia-smi 
# Check if the artifacts can be built
cd /work/Art/build
make -j

# The expected output is:
# [  5%] Built target test-plaintext-gcn-small
# [ 11%] Built target test-plaintext-gcn # The executable for plaintext global training
# [ 17%] Built target test-fed-gcn # The executable for FL (FedAvg)-based approach
# [ 40%] Built target TaskHandler
# [ 45%] Built target test-fhe-wrapper
# [ 51%] Built target test-comm
# [ 57%] Built target gcn-inference-optimize # The executable for optimized GCN inference
# [ 62%] Built target test-server
# [ 68%] Built target test-client
# [ 74%] Built target gcn-ss 
# [ 80%] Built target test-2PC # The executable for 2PC unit test
# [ 88%] Built target test-graphsc # The executale for GraphSC (SML-based state-of-the-art)
# [ 94%] Built target gcn-original # The executale for unoptimized GCN training
# [100%] Built target gcn-optimize # The executale for optimized GCN training
```

Run a smallest training test (~1 min):
- The smallest training test corresponds to one specific setting in our efficiency evaluation, i.e., 2-party training, Cora dataset, 2 epochs with preprocessing. See *Section 7.2.1 Setup-Dataset* of our paper for how efficiency evaluation datasets are set up. 
- This might not be as fast as what we measured on the host machine due to the virtualization of Docker (as well as your hardware resources). 
```bash
cd /work/Art/CoGNN/tools
python tmp_run_cluster.py --smallest-cognn-efficiency

# The expected console output is:
# ip netns exec A ./../../bin/gcn-optimize -t 2 -g 2 -i 0 -m 12 -p 1 -s gcn-optimize/cora/2s -c 1 -r 1 ./data/Cora/transformed/2s/cora.edge.preprocessed ./data/Cora/transformed/2s/cora.vertex.preprocessed ./data/Cora/transformed/2s/cora.part.preprocessed ./cognn-smallest/result/gcn-optimize/cora/2s/gcn_test_0.result.cora ./data/Cora/transformed/2s/cora_config.txt
# ip netns exec B ./../../bin/gcn-optimize -t 2 -g 2 -i 1 -m 12 -p 1 -s gcn-optimize/cora/2s -c 1 -r 1 ./data/Cora/transformed/2s/cora.edge.preprocessed ./data/Cora/transformed/2s/cora.vertex.preprocessed ./data/Cora/transformed/2s/cora.part.preprocessed ./cognn-smallest/result/gcn-optimize/cora/2s/gcn_test_1.result.cora ./data/Cora/transformed/2s/cora_config.txt
```

The two command lines correspond to two parties (processes). And you will get three new folders under the `tools/` directory, as follows:
```bash
└── 📁tools
    └── 📁data
    └── 📁plot
    └── 📁scripts
    └── 📁ot-data # OT correlations for Ferret-OT (EMP)
    └── 📁preprocess # Preprocessed OEP correlations
    └── 📁cognn-smallest # The output folder for cognn-smallest test
        └── 📁comm # Communication of each party
        └── 📁log # Log of each party, recording detailed running durations and accuracies of each epoch
        └── 📁result # Output of each party, currently containing nothing since we did not reconstruct the secret shares       
    └── tmp_run_cluster.py
```
Feel free to browse files (especially the log files) under these folders to better understand the running process of CoGNN. For example:
```bash
root@<container-id>:/work/Art/CoGNN/tools/cognn-smallest/log/gcn-optimize/cora/2s# cat gcn_test_cora_0.log | grep accuracy
# full set accuracy = 19.188192
# training set accuracy = 20.370370
# border training set accuracy = 22.916667
# test set accuracy = 20.552147
# border test set accuracy = 21.556886
# full set accuracy = 24.907749
# training set accuracy = 28.703704
# border training set accuracy = 33.333333
# test set accuracy = 26.073620
# border test set accuracy = 25.748503
root@<container-id>:/work/Art/CoGNN/tools/cognn-smallest/log/gcn-optimize/cora/2s# cat gcn_test_cora_0.log | grep "preprocess took"
# ::preprocess took 7.611 seconds
root@<container-id>:/work/Art/CoGNN/tools/cognn-smallest/log/gcn-optimize/cora/2s# cat gcn_test_cora_0.log | grep "iteration took"
# ::iteration took 4.124 seconds
# ::iteration took 5.253 seconds
# ::iteration took 0.374 seconds
# ::iteration took 1.779 seconds
# ::iteration took 0.429 seconds
# ::iteration took 7.674 seconds
# ::iteration took 4.329 seconds
# ::iteration took 5.428 seconds
# ::iteration took 0.444 seconds
# ::iteration took 1.877 seconds
# ::iteration took 0.407 seconds
# ::iteration took 9.482 seconds
root@<container-id>:/work/Art/CoGNN/tools/cognn-smallest/comm# cat Truepreprocess_1scaler_gcn-optimize_cora_2p.comm 
#         iface   Download     Upload
# 2  vethA.peer  1237.18MB  1215.83MB
# 3  vethB.peer  1215.83MB  1237.18MB
# 0          lo     0.00MB     0.00MB
# 1         br0     0.00MB     0.00MB
# 4  vethC.peer     0.00MB     0.00MB
# 5  vethD.peer     0.00MB     0.00MB
# 6  vethE.peer     0.00MB     0.00MB
# 7        eth0     0.00MB     0.00MB
```

## 4 Evaluation

Now let's head for the full evaluations corresponding to the key results obtained in our paper. Fully running all the experiments in our paper might **take 3 days or more**. You can selectively verify some specific settings.

**Cautions:**
- After running each part of the evaluation, you'd better **clean the `preprocess/` folder**. Otherwise you disk space would soon be consumed up.
- **DO NOT** clean the log and comm folders, since they would be used for plot.

Set up a container and cd to our evaluation scripts:
```bash
# Enter the container first. Please select the proper tagname for yourself.
sudo docker run -it --rm --privileged --security-opt apparmor=unconfined --gpus all cbackyx/cognn-ae:<tagname> /bin/bash

# Now you are in the bash of your container. cd to our evaluation scripts.
cd /work/Art/CoGNN/tools/
```

The evaluation options provided by `tmp_run_cluster.py` include:
> Note that we also specify which option (setting) corresponds to which Figure/Table in our paper. 
```bash
python tmp_run_cluster.py -h
# usage: tmp_run_cluster.py [-h] [--cognn-opt-accuracy] [--fedgnn-accuracy] [--plaintextgnn-accuracy] [--graphsc-efficiency] [--cognn-opt-efficiency] [--cognn-unopt-efficiency] [--cognn-opt-inference] [--cognn-unopt-inference] [--smallest-cognn-efficiency]

# Evaluate CoGNN and GraphSC models.

# optional arguments:
#   -h, --help            show this help message and exit
#   --cognn-opt-accuracy  Evaluate CoGNN-Opt accuracy (~16h, Figure 7, Table 1, Table 9)
#   --cognn-unopt-accuracy  Evaluate CoGNN-Opt effiency under the accuracy setting (~6h, Table 7, Table 9)
#   --fedgnn-accuracy     Evaluate FL-based GNN accuracy (~30min, Figure 7)
#   --plaintextgnn-accuracy
#                         Evaluate Plaintext GNN accuracy (~1h, Figure 7)
#   --graphsc-efficiency  Evaluate GraphSC efficiency (~20h, Figure 6)
#   --cognn-opt-efficiency
#                         Evaluate CoGNN-Opt efficiency (~1h, Figure 6)
#   --cognn-unopt-efficiency
#                         Evaluate CoGNN unoptimized efficiency (~8h, Figure 6)
#   --cognn-opt-inference
#                         Evaluate CoGNN-Opt inference efficiency (~10min, Table 2, Table 10)
#   --cognn-unopt-inference
#                         Evaluate CoGNN unoptimized inference efficiency (~6h, Table 8, Table 10)
#   --smallest-cognn-efficiency
#                         Evaluate smallest CoGNN efficiency (~1min)
```

See the function annotations in `tmp_run_cluster.py` for the detailed information on each evaluation setting.

You can `cat` the corresponding log files for each evaluation setting to view the current running progress.

As you have run all the experiments listed above, plot the results:

```bash
cd /work/Art/CoGNN/tools/plot
python plot_duration_and_comm_scale.py # Figure 6
python plot_multiparty_accuracy.py # Figure 7, Table 11, 12
python plot_duration_breakdown_and_comm.py # Table 1, 2, 7, 8, 9, 10
python plot_message_passing_comm.py # Table 6
```

-----

The following scripts tells how to set up dependencies of CoGNN and CoGNN itself from scratch. They are not recommended!

## -1 Set up the dependencies

### -1.1 Environment Requirement

- Ubuntu 20.04
- g++/gcc 9.4.0
- GNU Make 4.2.1
- cmake 3.25.1
- Python 3.9.13
- CUDA Driver Version: 535.113.01  CUDA Version: 12.2

### -1.2 Set up the workspace

```bash
mkdir Artifact && cd Artifact
mkdir Art
```

### -1.3 Install MPC dependencies.

Install EMP.

```bash
cd Artifact
git clone https://github.com/CoGNN-anon/MPC.git
cd MPC
python install.py --deps --tool --ot --sh2pc
```

Install libOTe.

```bash
cd Artifact/Art/
git clone https://github.com/osu-crypto/libOTe.git
cd libOTe
git checkout 3a40823
git submodule update --init
# Open the serialization module for boost, in 'libOTe/cryptoTools/thirdparty/getBoost.py'. Then:
python build.py --setup --boost --relic
python build.py --install=./../lilibOTe -D ENABLE_RELIC=ON -D ENABLE_NP=ON -D ENABLE_KOS=ON -D ENABLE_IKNP=ON -D ENABLE_OOS=ON -D ENABLE_SILENTOT=ON
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
cmake .. -DCMAKE_INSTALL_PREFIX=./install
make -j
cmake --build . --target install --parallel
```

Install troy (GPU-based FHE).

```bash
cd Artifact/MPC
git clone https://github.com/CoGNN-anon/troy.git
cd troy
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=./install
make -j
cmake --build . --target install --parallel
```

### -1.4 Set Up CMake Dependencies

```bash
cd Artifact/MPC
sudo cp -r cmake/* /usr/local/lib/cmake/
```

## -2 Set up CoGNN

> Note that, you might have to manually set some paths in the CMakeList.txt according to your environment.

```bash
cd Artifact/Art/

git clone https://github.com/CoGNN-anon/CoGNN.git
git clone https://github.com/CoGNN-anon/Task-Worker.git

mkdir build && cd build
cmake .. -DTHREADING=ON
make -j8
```

## -3 Prepare the datasets

```bash
cd Artifact/Art/CoGNN/tools
python data_transform.py
```

## -4 Run evaluation

```bash
cd Artifact/Art/CoGNN/tools
python tmp_run_cluster.py
```


