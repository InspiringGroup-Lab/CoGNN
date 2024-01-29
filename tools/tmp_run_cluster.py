import subprocess
import sys
import os
from threading import Thread
import psutil
import time
import pandas as pd
import shutil

executable_root_path = "./../../bin/"
data_root_path = "./data/"
ferret_ot_data_root_path = "./ot-data/"
result_root_path = "./result/"
log_root_path = "./log/"
config_root_path = "./config/"
preprocess_root_path = "./preprocess/"
comm_root_path = "./comm/"
# iterations = 120
iterations = 4
defaultNumParts = 2
doPreprocess = False
defaultBandWidth = 400 #Mbps
defaultLatency = 1 #ms
isCluster = True
defaultInterRatio = 0.4
defaultScaler = 0.2
bandwidthList = [200, 400, 1000, 4000]
latencyList = [0.15, 1, 10, 20]

def my_makedir(path):
    if not os.path.isdir(path):
        os.makedirs(path)

def delete_and_create_dir(dir_path):
    # check if the directory already exists
    if os.path.exists(dir_path):
        # delete the existing directory
        shutil.rmtree(dir_path)
    # create a new directory
    os.makedirs(dir_path)

# print(process.stdout.read().decode('utf-8'))

# python batch_Deckard.py > test_case/batch_similarity.txt

# ./sssp-ss -t 5 -g 5 -i 4 -m 2 -p 1 ./../data/test.dat ./../data/test5p.part ./../data/result_4.txt 0

processList = []

def setup_network(bandwidth, latency):
    # Clean first
    curProc = subprocess.Popen(["sudo", "bash", "./scripts/clean_network.sh"])
    curProc.wait()
    # Setup network
    curProc = subprocess.Popen(["sudo", "bash", "./scripts/setup_network.sh", str(bandwidth), str(latency)])
    curProc.wait()

def clean_network():
    curProc = subprocess.Popen(["sudo", "bash", "./scripts/clean_network.sh"])
    curProc.wait()

nsList = ["A", "B", "C", "D", "E"]

def get_size(bytes):
    """
    Returns size of bytes in a nice format
    """
    # for unit in ['', 'K', 'M', 'G', 'T', 'P']:
    #     if bytes < 1024:
    #         return f"{bytes:.2f}{unit}B"
    #     bytes /= 1024
    bytes /= 1024
    bytes /= 1024
    return f"{bytes:.2f}MB"

def commStart():
    io = psutil.net_io_counters(pernic=True)
    return io

def commEnd(io, doPreprocess, scaler, tag=""):
    # get the network I/O stats again per interface 
    io_2 = psutil.net_io_counters(pernic=True)
    # initialize the data to gather (a list of dicts)
    data = []
    for iface, iface_io in io.items():
        # new - old stats gets us the speed
        upload, download = io_2[iface].bytes_sent - iface_io.bytes_sent, io_2[iface].bytes_recv - iface_io.bytes_recv
        data.append({
            "iface": iface, "Download": get_size(download),
            "Upload": get_size(upload),
        })
    # construct a Pandas DataFrame to print stats in a cool tabular style
    df = pd.DataFrame(data)
    # sort values per column, feel free to change the column
    df.sort_values("Download", inplace=True, ascending=False)
    # clear the screen based on your OS
    # os.system("cls") if "nt" in os.name else os.system("clear")
    # print the stats
    # print(df.to_string())
    my_makedir(comm_root_path)
    with open(comm_root_path + str(doPreprocess) + "preprocess_" + str(scaler) + "scaler" + tag + ".comm", "w", encoding='utf-8') as ofile:
        ofile.write(df.to_string())

def run_gcn_test(executable = "gcn-ss", dataset = "cora", numParts = 2):
    dataset_upper_case = dataset[:1].upper() + dataset[1:]
    cur_bandwidth = 4000
    cur_latency = defaultLatency
    if isCluster:
        setup_network(cur_bandwidth, cur_latency)
    # Evaluation root setting
    data_path = data_root_path + "gcn_test/"
    delete_and_create_dir(ferret_ot_data_root_path)
    result_path = result_root_path + executable + "/" + dataset + "/" + str(numParts) + "p/"
    if doPreprocess:
        log_path = log_root_path + executable + "/" + dataset + "/" + str(numParts) + "p/"
    else:
        log_path = log_root_path + executable + "/" + dataset + "/" + str(numParts) + "p/" + "noPreprocess/"
    my_makedir(result_path)
    my_makedir(log_path)
    preprocess_path = preprocess_root_path + executable + "/" + dataset + "/" + str(numParts) + "p/"
    my_makedir(preprocess_path)
    
    preprocess_setting = executable + "/" + dataset + "/" + str(numParts) + "p"
    executable_path = executable_root_path + executable
    processList = []
    start_io = commStart()
    for i in range(numParts):
        cmd = []
        if isCluster:
            cmd = ["sudo", "ip", "netns", "exec", nsList[i]]
        cmd += [executable_path, "-t", str(numParts), "-g", str(numParts), "-i", str(i), "-m", str(iterations), "-p", "1", "-s", preprocess_setting]
        if not doPreprocess:
            cmd += ["-n", "1"]
        if isCluster:
            cmd += ["-c", "1"]
        cmd += ["-r", "1"]
        # cmd += ["/home/zzh/project/test-GCN/FedGCNData/data/cora/cora.edge.small", "/home/zzh/project/test-GCN/FedGCNData/data/cora/cora.vertex.small",
        #         "/home/zzh/project/test-GCN/FedGCNData/data/cora/cora.part.small", "./result/gcn_test/gcn_test"+"_"+str(i)+".result"]
        # cmd += ["/home/zzh/project/test-GCN/FedGCNData/data/tiny/cora.edge.small", "/home/zzh/project/test-GCN/FedGCNData/data/tiny/cora.vertex.small",
        #         "/home/zzh/project/test-GCN/FedGCNData/data/tiny/cora.part.small", "./result/gcn_test/gcn_test"+"_"+str(i)+".result"]

        # cmd += ["/home/zzh/project/test-GCN/FedGCNData/data/" + dataset + "/" + dataset + ".edge.preprocessed", 
        #         "/home/zzh/project/test-GCN/FedGCNData/data/" + dataset + "/" + dataset + ".vertex.preprocessed",
        #         "/home/zzh/project/test-GCN/FedGCNData/data/" + dataset + "/" + dataset + ".part.preprocessed." + str(numParts) + "p", 
        #         result_path + "gcn_test"+"_"+str(i)+".result." + dataset,
        #         config_root_path + dataset + "_config.txt"]

        cmd += [f"/home/zzh/project/SecGNN/data/{dataset_upper_case}/transformed/" + dataset + ".edge.preprocessed", 
                f"/home/zzh/project/SecGNN/data/{dataset_upper_case}/transformed/" + dataset + ".vertex.preprocessed",
                f"/home/zzh/project/SecGNN/data/{dataset_upper_case}/transformed/" + dataset + ".part.preprocessed." + str(numParts) + "p", 
                result_path + "gcn_test"+"_"+str(i)+".result." + dataset,
                f"/home/zzh/project/SecGNN/data/{dataset_upper_case}/transformed/" + dataset + "_config.txt"]

        print(" ".join(cmd))
        log_f = open(log_path+"gcn_test"+"_"+dataset+"_"+str(i)+".log", 'w', encoding='utf-8')
        processList.append(subprocess.Popen(cmd, stdout=log_f))
    for process in processList:
        process.wait()
    commEnd(start_io, doPreprocess, 1, str("_") + executable + "_" + dataset + "_" + str(numParts) + "p")
    processList = []

def run_graphsc(executable = "test-graphsc", dataset = "cora", scaler = 2):
    dataset_upper_case = dataset[:1].upper() + dataset[1:]
    cur_bandwidth = 4000
    cur_latency = defaultLatency

    if isCluster:
        setup_network(cur_bandwidth, cur_latency)
    # Evaluation root setting
    delete_and_create_dir(ferret_ot_data_root_path)
    result_path = result_root_path + executable + "/" + dataset + "/" + str(scaler) + "s/"
    if doPreprocess:
        log_path = log_root_path + executable + "/" + dataset + "/" + str(scaler) + "s/"
    else:
        log_path = log_root_path + executable + "/" + dataset + "/" + str(scaler) + "s/" + "noPreprocess/"
    my_makedir(result_path)
    my_makedir(log_path)
    preprocess_path = preprocess_root_path + executable + "/" + dataset + "/" + str(scaler) + "s/"
    my_makedir(preprocess_path)
    
    preprocess_setting = executable + "/" + dataset + "/" + str(scaler) + "s"
    executable_path = executable_root_path + executable
    processList = []
    start_io = commStart()
    numParties = 2
    epochs = 1
    for i in range(numParties):
        cmd = []
        if isCluster:
            cmd = ["sudo", "ip", "netns", "exec", nsList[i]]
        # partyId GNNConfigFile v_path e_path setting n_epochs do_prep is_cluster
        cmd += [executable_path, \
                str(i), \
                f"/home/zzh/project/SecGNN/data/{dataset_upper_case}/transformed/{scaler}s/" + dataset + "_config.txt", \
                f"/home/zzh/project/SecGNN/data/{dataset_upper_case}/transformed/{scaler}s/" + dataset + ".vertex.preprocessed", \
                f"/home/zzh/project/SecGNN/data/{dataset_upper_case}/transformed/{scaler}s/" + dataset + ".edge.preprocessed", \
                preprocess_setting, \
                str(epochs), \
                str(1) if doPreprocess else str(0), \
                str(1) if isCluster else str(0)]
        print(" ".join(cmd))
        # continue
        log_f = open(log_path+"gcn_test"+"_"+dataset+"_"+str(i)+".log", 'w', encoding='utf-8')
        processList.append(subprocess.Popen(cmd, stdout=log_f))
    for process in processList:
        process.wait()
    commEnd(start_io, doPreprocess, 1, str("_") + executable + "_" + dataset + "_" + str(scaler) + "s")
    processList = []

def run_cognn_scaler(executable = "gcn-ss", dataset = "cora", numParts = 2):
    dataset_upper_case = dataset[:1].upper() + dataset[1:]
    cur_bandwidth = 4000
    cur_latency = defaultLatency
    if isCluster:
        setup_network(cur_bandwidth, cur_latency)
    # Evaluation root setting
    data_path = data_root_path + "gcn_test/"
    delete_and_create_dir(ferret_ot_data_root_path)
    result_path = result_root_path + executable + "/" + dataset + "/" + str(numParts) + "s/"
    if doPreprocess:
        log_path = log_root_path + executable + "/" + dataset + "/" + str(numParts) + "s/"
    else:
        log_path = log_root_path + executable + "/" + dataset + "/" + str(numParts) + "s/" + "noPreprocess/"
    my_makedir(result_path)
    my_makedir(log_path)
    preprocess_path = preprocess_root_path + executable + "/" + dataset + "/" + str(numParts) + "s/"
    my_makedir(preprocess_path)
    
    preprocess_setting = executable + "/" + dataset + "/" + str(numParts) + "s"
    executable_path = executable_root_path + executable
    processList = []
    start_io = commStart()
    for i in range(numParts):
        cmd = []
        if isCluster:
            cmd = ["sudo", "ip", "netns", "exec", nsList[i]]
        cmd += [executable_path, "-t", str(numParts), "-g", str(numParts), "-i", str(i), "-m", str(iterations), "-p", "1", "-s", preprocess_setting]
        if not doPreprocess:
            cmd += ["-n", "1"]
        if isCluster:
            cmd += ["-c", "1"]
        cmd += ["-r", "1"]
        # cmd += ["/home/zzh/project/test-GCN/FedGCNData/data/cora/cora.edge.small", "/home/zzh/project/test-GCN/FedGCNData/data/cora/cora.vertex.small",
        #         "/home/zzh/project/test-GCN/FedGCNData/data/cora/cora.part.small", "./result/gcn_test/gcn_test"+"_"+str(i)+".result"]
        # cmd += ["/home/zzh/project/test-GCN/FedGCNData/data/tiny/cora.edge.small", "/home/zzh/project/test-GCN/FedGCNData/data/tiny/cora.vertex.small",
        #         "/home/zzh/project/test-GCN/FedGCNData/data/tiny/cora.part.small", "./result/gcn_test/gcn_test"+"_"+str(i)+".result"]
        cmd += [f"/home/zzh/project/SecGNN/data/{dataset_upper_case}/transformed/{numParts}s/" + dataset + ".edge.preprocessed", 
                f"/home/zzh/project/SecGNN/data/{dataset_upper_case}/transformed/{numParts}s/" + dataset + ".vertex.preprocessed",
                f"/home/zzh/project/SecGNN/data/{dataset_upper_case}/transformed/{numParts}s/" + dataset + ".part.preprocessed",
                result_path + "gcn_test"+"_"+str(i)+".result." + dataset,
                f"/home/zzh/project/SecGNN/data/{dataset_upper_case}/transformed/{numParts}s/" + dataset + "_config.txt"]
        print(" ".join(cmd))
        log_f = open(log_path+"gcn_test"+"_"+dataset+"_"+str(i)+".log", 'w', encoding='utf-8')
        processList.append(subprocess.Popen(cmd, stdout=log_f))
    for process in processList:
        process.wait()
    commEnd(start_io, doPreprocess, 1, str("_") + executable + "_" + dataset + "_" + str(numParts) + "p")
    processList = []

def set_root_paths(application):
    global executable_root_path, data_root_path, ferret_ot_data_root_path, result_root_path, log_root_path, config_root_path, preprocess_root_path, comm_root_path
    if application == "cognn-scale":
        executable_root_path = f"./../../bin/"
    elif application == "graphsc":
        executable_root_path = f"./../../build/bin/"
    data_root_path = f"./{str(application)}/data/"
    ferret_ot_data_root_path = f"./{str(application)}/ot-data/"
    result_root_path = f"./{str(application)}/result/"
    log_root_path = f"./{str(application)}/log/"
    config_root_path = "./config/"
    preprocess_root_path = f"./preprocess/"
    comm_root_path = f"./{str(application)}/comm/"

# doPreprocess = True
# # run_gcn_test("gcn-optimize", "citeseer")
# # run_gcn_test("gcn-optimize", "pubmed", 2)
# # run_gcn_test("gcn-ss", "cora", 3)
# # run_gcn_test("gcn-optimize", "cora", 3)
# # run_gcn_test("gcn-original", "cora")

# list_num_parts = [2]
# iterations = 540
# for cur_num_parts in list_num_parts:
#     # run_gcn_test("gcn-optimize", "cora", cur_num_parts)
#     # run_gcn_test("gcn-optimize", "citeseer", cur_num_parts)
#     # run_gcn_test("gcn-optimize", "pubmed", cur_num_parts)
# # iterations = 4
# # for cur_num_parts in list_num_parts:
# #     run_gcn_test("gcn-original", "cora", cur_num_parts)
# #     run_gcn_test("gcn-original", "citeseer", cur_num_parts)
# #     run_gcn_test("gcn-original", "pubmed", cur_num_parts)

# set_root_paths("mp-accuracy")
# doPreprocess = False
# iterations = 540
# list_num_parts = [5, 4, 3, 2]
# for cur_num_parts in list_num_parts:
#     # run_gcn_test("gcn-optimize", "cora", cur_num_parts)
#     # run_gcn_test("gcn-optimize", "citeseer", cur_num_parts)
#     run_gcn_test("gcn-optimize", "pubmed", cur_num_parts)

# list_num_parts = [2]
# list_datasets = ["cora", "citeseer", "pubmed"]
# iterations = 2

# doPreprocess = True
# for cur_num_parts in list_num_parts:
#     for cur_dataset in list_datasets:
#         run_gcn_test("gcn-inference-optimize", cur_dataset, cur_num_parts)

# doPreprocess = False
# for cur_num_parts in list_num_parts:
#     for cur_dataset in list_datasets:
#         run_gcn_test("gcn-inference-optimize", cur_dataset, cur_num_parts)

# >>>> test graphsc scale
 
# set_root_paths("graphsc")
# # list_datasets = ["cora", "citeseer", "pubmed"]
# list_datasets = ["cora", "citeseer", "pubmed"]
# # list_scalers = [2, 5]
# # doPreprocess = True
# # for dataset in list_datasets:
# #     for cur_scaler in list_scalers:
# #         if cur_scaler == 5 and dataset == "citeseer":
# #             continue
# #         run_graphsc("test-graphsc", dataset, cur_scaler)

# list_scalers = [2, 3, 4, 5]
# doPreprocess = False
# for dataset in list_datasets:
#     for cur_scaler in list_scalers:
#         run_graphsc("test-graphsc", dataset, cur_scaler)
    
# # >>>> test cognn scale

# list_scalers = [2, 3, 4, 5]
# iterations = 6
# list_datasets = ["cora", "citeseer", "pubmed"]

# set_root_paths("cognn-scale")
# # doPreprocess = True
# # for dataset in list_datasets:
# #     for cur_scaler in list_scalers:
# #         run_cognn_scaler("gcn-optimize", dataset, cur_scaler)

# doPreprocess = False
# for dataset in list_datasets:
#     for cur_scaler in list_scalers:
#         run_cognn_scaler("gcn-optimize", dataset, cur_scaler)

# list_scalers = [2, 3, 4, 5]
# iterations = 4
# list_datasets = ["cora", "citeseer", "pubmed"]

# set_root_paths("cognn-scale")
# # doPreprocess = True
# # for dataset in list_datasets:
# #     for cur_scaler in list_scalers:
# #         run_cognn_scaler("gcn-original", dataset, cur_scaler)

# doPreprocess = False
# for dataset in list_datasets:
#     for cur_scaler in list_scalers:
#         run_cognn_scaler("gcn-original", dataset, cur_scaler)

# set_root_paths("mp-accuracy")
# doPreprocess = False
# iterations = 540
# list_datasets = ["citeseer"]
# for dataset in list_datasets:
#     run_gcn_test("gcn-optimize", dataset, 2)

# iterations = 4
# doPreprocess = True
# list_datasets = ["cora", "pubmed", "citeseer"]
# for dataset in list_datasets:
#     run_gcn_test("gcn-original", dataset, 2)
# doPreprocess = False
# for dataset in list_datasets:
#     run_gcn_test("gcn-original", dataset, 2)

set_root_paths("inference")

list_num_parts = [2]
list_datasets = ["cora", "citeseer", "pubmed"]
iterations = 2

doPreprocess = True
for cur_num_parts in list_num_parts:
    for cur_dataset in list_datasets:
        run_gcn_test("gcn-inference-optimize", cur_dataset, cur_num_parts)
        run_gcn_test("gcn-original", cur_dataset, cur_num_parts)

doPreprocess = False
for cur_num_parts in list_num_parts:
    for cur_dataset in list_datasets:
        run_gcn_test("gcn-inference-optimize", cur_dataset, cur_num_parts)
        run_gcn_test("gcn-original", cur_dataset, cur_num_parts)
