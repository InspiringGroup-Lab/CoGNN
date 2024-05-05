import subprocess
import sys
import os
from threading import Thread
import psutil
import time
import pandas as pd
import shutil
import argparse

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
    curProc = subprocess.Popen(["bash", "./scripts/clean_network.sh"])
    curProc.wait()
    # Setup network
    curProc = subprocess.Popen(["bash", "./scripts/setup_network.sh", str(bandwidth), str(latency)])
    curProc.wait()

def clean_network():
    curProc = subprocess.Popen(["bash", "./scripts/clean_network.sh"])
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
            cmd = ["ip", "netns", "exec", nsList[i]]
        cmd += [executable_path, "-t", str(numParts), "-g", str(numParts), "-i", str(i), "-m", str(iterations), "-p", "1", "-s", preprocess_setting]
        if not doPreprocess:
            cmd += ["-n", "1"]
        if isCluster:
            cmd += ["-c", "1"]
        cmd += ["-r", "1"]

        cmd += [f"./data/{dataset_upper_case}/transformed/" + dataset + ".edge.preprocessed", 
                f"./data/{dataset_upper_case}/transformed/" + dataset + ".vertex.preprocessed",
                f"./data/{dataset_upper_case}/transformed/" + dataset + ".part.preprocessed." + str(numParts) + "p", 
                result_path + "gcn_test"+"_"+str(i)+".result." + dataset,
                f"./data/{dataset_upper_case}/transformed/" + dataset + "_config.txt"]

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
            cmd = ["ip", "netns", "exec", nsList[i]]
        # partyId GNNConfigFile v_path e_path setting n_epochs do_prep is_cluster
        cmd += [executable_path, \
                str(i), \
                f"./data/{dataset_upper_case}/transformed/{scaler}s/" + dataset + "_config.txt", \
                f"./data/{dataset_upper_case}/transformed/{scaler}s/" + dataset + ".vertex.preprocessed", \
                f"./data/{dataset_upper_case}/transformed/{scaler}s/" + dataset + ".edge.preprocessed", \
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
            cmd = ["ip", "netns", "exec", nsList[i]]
        cmd += [executable_path, "-t", str(numParts), "-g", str(numParts), "-i", str(i), "-m", str(iterations), "-p", "1", "-s", preprocess_setting]
        if not doPreprocess:
            cmd += ["-n", "1"]
        if isCluster:
            cmd += ["-c", "1"]
        cmd += ["-r", "1"]
        cmd += [f"./data/{dataset_upper_case}/transformed/{numParts}s/" + dataset + ".edge.preprocessed", 
                f"./data/{dataset_upper_case}/transformed/{numParts}s/" + dataset + ".vertex.preprocessed",
                f"./data/{dataset_upper_case}/transformed/{numParts}s/" + dataset + ".part.preprocessed",
                result_path + "gcn_test"+"_"+str(i)+".result." + dataset,
                f"./data/{dataset_upper_case}/transformed/{numParts}s/" + dataset + "_config.txt"]
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
    # ferret_ot_data_root_path = f"./{str(application)}/ot-data/"
    result_root_path = f"./{str(application)}/result/"
    log_root_path = f"./{str(application)}/log/"
    config_root_path = "./config/"
    preprocess_root_path = f"./preprocess/"
    comm_root_path = f"./{str(application)}/comm/"

# Evaluate the accuracy of CoGNN-Opt for the three datasets and for different numbers of parties. Trained for 90 epochs under each setting. 
# Note that, every 6 iterations correspond to 1 epoch of CoGNN-Opt, where the former 2 (GAS) iterations are in a forward pass and the latter 4 iterations are in a backward pass. 
def eval_cognn_opt_accuracy():
    global doPreprocess, iterations
    set_root_paths("mp-accuracy")
    doPreprocess = True
    iterations = 540
    list_num_parts = [2,3,4,5]
    for cur_num_parts in list_num_parts:
        run_gcn_test("gcn-optimize", "cora", cur_num_parts)
        run_gcn_test("gcn-optimize", "citeseer", cur_num_parts)
        run_gcn_test("gcn-optimize", "pubmed", cur_num_parts)

def eval_cognn_opt_accuracy_no_preprocess():
    global doPreprocess, iterations
    set_root_paths("mp-accuracy")
    doPreprocess = False
    iterations = 540
    list_num_parts = [2]
    for cur_num_parts in list_num_parts:
        run_gcn_test("gcn-optimize", "cora", cur_num_parts)
        run_gcn_test("gcn-optimize", "citeseer", cur_num_parts)
        run_gcn_test("gcn-optimize", "pubmed", cur_num_parts)

# Evaluate the efficiency of CoGNN (under the efficiency setting for full graph computation) for the three datasets and for 2 parties. Trained for 1 epoch under each setting. 
# Note that, every 4 iterations correspond to 1 epoch of CoGNN, where the former 2 (GAS) iterations are in a forward pass and the latter 2 iterations are in a backward pass. 
def eval_cognn_unopt_accuracy():
    global doPreprocess, iterations
    set_root_paths("mp-accuracy")
    doPreprocess = True
    iterations = 4
    list_num_parts = [2]
    for cur_num_parts in list_num_parts:
        run_gcn_test("gcn-original", "cora", cur_num_parts)
        run_gcn_test("gcn-original", "citeseer", cur_num_parts)
        run_gcn_test("gcn-original", "pubmed", cur_num_parts)

def eval_cognn_unopt_accuracy_no_preprocess():
    global doPreprocess, iterations
    set_root_paths("mp-accuracy")
    doPreprocess = False
    iterations = 4
    list_num_parts = [2]
    for cur_num_parts in list_num_parts:
        run_gcn_test("gcn-original", "cora", cur_num_parts)
        run_gcn_test("gcn-original", "citeseer", cur_num_parts)
        run_gcn_test("gcn-original", "pubmed", cur_num_parts)

def eval_fedgnn_accuracy():
    set_root_paths("mp-accuracy")
    list_num_parts = [2,3,4,5]
    list_datasets = ["cora", "citeseer", "pubmed"]
    log_path = log_root_path
    my_makedir(log_path)
    processList = []
    for dataset in list_datasets:
        for cur_num_parts in list_num_parts:
            executable_path = "/work/Art/build/bin/test-fed-gcn"
            cmd = [executable_path, str(dataset), str(cur_num_parts)]
            print(" ".join(cmd))
            log_f = open(log_path+"fed-gcn."+dataset+"."+str(cur_num_parts)+"p.log", 'w', encoding='utf-8')
            processList.append(subprocess.Popen(cmd, stdout=log_f))
    for process in processList:
        process.wait()

def eval_plaintextgnn_accuracy():
    set_root_paths("mp-accuracy")
    list_datasets = ["cora", "citeseer", "pubmed"]
    log_path = log_root_path
    my_makedir(log_path)
    processList = []
    for dataset in list_datasets:
        executable_path = "/work/Art/build/bin/test-plaintext-gcn"
        cmd = [executable_path, str(dataset)]
        print(" ".join(cmd))
        log_f = open(log_path+"plaintext-gcn."+dataset+".log", 'w', encoding='utf-8')
        processList.append(subprocess.Popen(cmd, stdout=log_f))
    for process in processList:
        process.wait()
 
# Evaluate the training efficiency of GraphSC for the three datasets and for different numbers of parties. Trained for 1 epoch under each setting. 
# Note that, every 4 iterations correspond to 1 epoch of GraphSC, where the former 2 (GAS) iterations are in a forward pass and the latter 2 iterations are in a backward pass. 
def eval_graphsc_efficiency():
    global iterations, doPreprocess
    set_root_paths("graphsc")
    list_datasets = ["cora", "citeseer", "pubmed"]
    list_scalers = [2, 3, 4, 5]
    doPreprocess = True
    for dataset in list_datasets:
        for cur_scaler in list_scalers:
            run_graphsc("test-graphsc", dataset, cur_scaler)

    list_scalers = [2, 3, 4, 5]
    doPreprocess = False
    for dataset in list_datasets:
        for cur_scaler in list_scalers:
            run_graphsc("test-graphsc", dataset, cur_scaler)
    
# Evaluate the training efficiency of CoGNN-Opt for the three datasets and for different numbers of parties. Trained for 1 epoch under each setting. 
# Note that, every 6 iterations correspond to 1 epoch of CoGNN-Opt, where the former 2 (GAS) iterations are in a forward pass and the latter 4 iterations are in a backward pass. 
def eval_cognn_opt_efficiency():
    global iterations, doPreprocess
    list_scalers = [2, 3, 4, 5]
    iterations = 6
    list_datasets = ["cora", "citeseer", "pubmed"]

    set_root_paths("cognn-scale")
    doPreprocess = True
    for dataset in list_datasets:
        for cur_scaler in list_scalers:
            run_cognn_scaler("gcn-optimize", dataset, cur_scaler)

    doPreprocess = False
    for dataset in list_datasets:
        for cur_scaler in list_scalers:
            run_cognn_scaler("gcn-optimize", dataset, cur_scaler)

# Evaluate the training efficiency of CoGNN for the three datasets and for different numbers of parties. Trained for 1 epoch under each setting. 
# Note that, every 4 iterations correspond to 1 epoch of CoGNN, where the former 2 (GAS) iterations are in a forward pass and the latter 2 iterations are in a backward pass.
def eval_cognn_unopt_efficiency():
    global iterations, doPreprocess
    list_scalers = [2, 3, 4, 5]
    iterations = 4
    list_datasets = ["cora", "citeseer", "pubmed"]

    set_root_paths("cognn-scale")
    doPreprocess = True
    for dataset in list_datasets:
        for cur_scaler in list_scalers:
            run_cognn_scaler("gcn-original", dataset, cur_scaler)

    doPreprocess = False
    for dataset in list_datasets:
        for cur_scaler in list_scalers:
            run_cognn_scaler("gcn-original", dataset, cur_scaler)

# Evaluate the inference efficiency of CoGNN-Opt for the three datasets and for 2 parties. Full-graph inference for 1 time under each setting. 
# Note that, every 2 iterations correspond to 1 inference of CoGNN-Opt.
def eval_cognn_opt_inference_efficiency():
    global iterations, doPreprocess
    set_root_paths("inference")

    list_num_parts = [2]
    list_datasets = ["cora", "citeseer", "pubmed"]
    iterations = 2

    doPreprocess = True
    for cur_num_parts in list_num_parts:
        for cur_dataset in list_datasets:
            run_gcn_test("gcn-inference-optimize", cur_dataset, cur_num_parts)

    doPreprocess = False
    for cur_num_parts in list_num_parts:
        for cur_dataset in list_datasets:
            run_gcn_test("gcn-inference-optimize", cur_dataset, cur_num_parts)

# Evaluate the inference efficiency of CoGNN for the three datasets and for 2 parties. Full-graph inference for 1 time under each setting. 
# Note that, every 2 iterations correspond to 1 inference of CoGNN.
def eval_cognn_unopt_inference_efficiency():
    global iterations, doPreprocess
    set_root_paths("inference")

    list_num_parts = [2]
    list_datasets = ["cora", "citeseer", "pubmed"]
    iterations = 2

    doPreprocess = True
    for cur_num_parts in list_num_parts:
        for cur_dataset in list_datasets:
            run_gcn_test("gcn-original", cur_dataset, cur_num_parts)

    doPreprocess = False
    for cur_num_parts in list_num_parts:
        for cur_dataset in list_datasets:
            run_gcn_test("gcn-original", cur_dataset, cur_num_parts)

# The smallest training test corresponds to one specific evaluation setting in our efficiency test, i.e., 2-party training, Cora dataset, 2 epochs with preprocessing.
def smallest_eval_cognn_efficiency():
    global iterations, doPreprocess
    list_scalers = [2]
    iterations = 12
    list_datasets = ["cora"]

    set_root_paths("cognn-smallest")
    doPreprocess = True
    for dataset in list_datasets:
        for cur_scaler in list_scalers:
            run_cognn_scaler("gcn-optimize", dataset, cur_scaler)

# Define the main function to parse command line arguments and call the appropriate functions
def main():
    parser = argparse.ArgumentParser(description='Evaluate CoGNN and GraphSC models.')
    parser.add_argument('--cognn-opt-accuracy', action='store_true', help='Evaluate CoGNN-Opt accuracy')
    parser.add_argument('--cognn-opt-accuracy-no-preprocess', action='store_true', help='Evaluate CoGNN-Opt accuracy, without preprocessing')
    parser.add_argument('--cognn-unopt-accuracy', action='store_true', help='Evaluate CoGNN unoptimized efficiency under the accuracy setting')
    parser.add_argument('--cognn-unopt-accuracy-no-preprocess', action='store_true', help='Evaluate CoGNN unoptimized efficiency under the accuracy setting, without preprocessing')
    parser.add_argument('--fedgnn-accuracy', action='store_true', help='Evaluate FL-based GNN accuracy')
    parser.add_argument('--plaintextgnn-accuracy', action='store_true', help='Evaluate Plaintext GNN accuracy')
    parser.add_argument('--graphsc-efficiency', action='store_true', help='Evaluate GraphSC efficiency')
    parser.add_argument('--cognn-opt-efficiency', action='store_true', help='Evaluate CoGNN-Opt efficiency')
    parser.add_argument('--cognn-unopt-efficiency', action='store_true', help='Evaluate CoGNN unoptimized efficiency')
    parser.add_argument('--cognn-opt-inference', action='store_true', help='Evaluate CoGNN-Opt inference efficiency')
    parser.add_argument('--cognn-unopt-inference', action='store_true', help='Evaluate CoGNN unoptimized inference efficiency')
    parser.add_argument('--smallest-cognn-efficiency', action='store_true', help='Evaluate smallest CoGNN efficiency')
    parser.add_argument('--all', action='store_true', help='Evaluate ALL')
    args = parser.parse_args()

    if args.cognn_opt_accuracy:
        eval_cognn_opt_accuracy()
    if args.cognn_opt_accuracy_no_preprocess:
        eval_cognn_opt_accuracy_no_preprocess()
    if args.cognn_unopt_accuracy:
        eval_cognn_unopt_accuracy()
    if args.cognn_unopt_accuracy_no_preprocess:
        eval_cognn_unopt_accuracy_no_preprocess()
    if args.fedgnn_accuracy:
        eval_fedgnn_accuracy()
    if args.plaintextgnn_accuracy:
        eval_plaintextgnn_accuracy()
    if args.graphsc_efficiency:
        eval_graphsc_efficiency()
    if args.cognn_opt_efficiency:
        eval_cognn_opt_efficiency()
    if args.cognn_unopt_efficiency:
        eval_cognn_unopt_efficiency()
    if args.cognn_opt_inference:
        eval_cognn_opt_inference_efficiency()
    if args.cognn_unopt_inference:
        eval_cognn_unopt_inference_efficiency()
    if args.smallest_cognn_efficiency:
        smallest_eval_cognn_efficiency()
    if args.all:
        eval_cognn_opt_accuracy()
        eval_cognn_opt_accuracy_no_preprocess()
        eval_cognn_unopt_accuracy()
        eval_cognn_unopt_accuracy_no_preprocess()
        eval_fedgnn_accuracy()
        eval_plaintextgnn_accuracy()
        eval_graphsc_efficiency()
        eval_cognn_opt_efficiency()
        eval_cognn_unopt_efficiency()
        eval_cognn_opt_inference_efficiency()
        eval_cognn_unopt_inference_efficiency()

if __name__ == "__main__":
    main()