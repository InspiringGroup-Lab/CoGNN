CoGNN
============

CoGNN is a provably secure, efficient and scalable framework for collaborative graph learning. It fully embraces distributed computation among all the participants (graph owners) and requires no centralized (trusted) server. CoGNN works in an in-place computation fashion and never outsources the raw gragh data of each participant, in order to honoring the raw data privacy. It is built upon 2-party secure computation backends, but can support involving an arbitrary number of participants. Currently, it supports the training and inference of GCN. 

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


