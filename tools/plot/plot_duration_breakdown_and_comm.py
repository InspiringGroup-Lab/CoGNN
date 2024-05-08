import matplotlib.pyplot as plt
from operator import add, sub
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import pandas as pd

# Define a function to sum up each consecutive num_iter_per_epoch durations
def sum_consecutive_durations(iter_duration, num_iter_per_epoch):
    # Initialize an empty list to store the sums
    sums = []
    
    # Loop through the iter_duration list with a step size of num_iter_per_epoch
    for i in range(0, len(iter_duration), num_iter_per_epoch):
        # Slice the iter_duration list from i to i + num_iter_per_epoch
        sub_list = iter_duration[i:i + num_iter_per_epoch]
        
        # Sum up the elements in the sub_list and append to the sums list
        sums.append(sum(sub_list))
    
    # Return the sums list
    return sums

def extract_cognn_durations(file_path, tag, num_epochs):
    # Arrays to store the durations
    duration = []
    
    # Open the file and read line by line
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for index, line in enumerate(lines):
            if '::' + tag + ' took' in line:
                # Extract and store the border test set accuracy
                # print(line)
                # cur_duration = float(line.split(' took ')[1].split(' ')[0].strip())
                try:
                    cur_duration = float(line.split(' took ')[1].split(' ')[0].strip())
                except:
                    cur_index = index
                    while True:
                        if 'seconds' in lines[cur_index]:
                            cur_duration = float(lines[cur_index].split(" seconds")[0].split(' ')[-1])
                            print(cur_duration)
                            break
                        cur_index += 1                    
                duration.append(cur_duration)
    
    return sum(duration) / num_epochs

# Define a function to process the communication file and return the average Download and Upload of each veth
def process_comm_file_and_get_veth_avg(file_path, num_epochs, num_parts):
    # Initialize an empty list to store the veth data
    veth_data = []
    
    # Open the file in read mode
    with open(file_path, 'r') as f:
        # Loop through each line in the file
        for line in f:
            # Split the line by whitespace
            line = line.split()
            
            # Check if the first element is a veth interface
            if line[1].startswith('veth'):
                # Convert the second and third elements to floats
                download = float(line[2].replace('MB', '').strip())
                upload = float(line[3].replace('MB', '').strip())
                
                # Append the download and upload to the veth data list
                veth_data.append([download, upload])
    
    # Initialize an empty list to store the sums
    sums = []

    # print(veth_data)
    
    # Loop through each pair of download and upload in the veth data list
    for download, upload in veth_data:
        # Sum up the download and upload and append to the sums list
        sums.append(download + upload)
    
    avg_veth = sum(sums) / num_parts
    avg_epoch = avg_veth / num_epochs / 2**10
    
    # Return the average
    return avg_epoch

datasets = ["cora", "citeseer", "pubmed"]
formatted_datasets = ["Cora", "CiteSeer", "PubMed"]
# datasets = ["cora", "citeseer"]
n_parts_list = [2]
n_epochs = 90
avg_epochs = 1
num_iter_per_epoch = 6
cognn_log_file = []
figure_root_path = "/work/Art/CoGNN/tools/plot/figure/multi-party/"
executable = "gcn-optimize"
cognn_log_root_path = "./../mp-accuracy/log/"
cognn_comm_root_path = "./../mp-accuracy/comm/"
preprocess_flag = "Truepreprocess"
duration_tag_list = ["preprocess", "PreScatterComp Client", "Scatter_preparation", "Scatter_computation", "premerging", "premerged_extraction", "Gather_computation", "Apply_computation"]
formatted_duration_tag_list = ["OM Pre", "PreScatter", "Scatter OM", "Scatter Comp", "Gather OG", "Gather OM", "Gather Comp", "Apply", "Total"]

def get_duration(n_parts, cur_preprocess_epochs, cur_online_epochs):
    cognn_duration = []
    for d in range(len(datasets)):
        dataset = datasets[d]
        cur_duration_list_addup = [0] * len(duration_tag_list)
        for i in range(n_parts):
            cur_log_file = cognn_log_root_path + executable + "/" + dataset + "/" + str(n_parts) + "p/" + "gcn_test_" + dataset + "_" + str(i) + ".log"
            cur_duration_list = []
            # print(cur_log_file)
            cur_duration_list.append(extract_cognn_durations(cur_log_file, duration_tag_list[0], cur_preprocess_epochs[d])) 
            for duration_tag in duration_tag_list[1:]:
                cur_duration = extract_cognn_durations(cur_log_file, duration_tag, cur_online_epochs)
                cur_duration_list.append(cur_duration)
            cur_duration_list_addup = list( map(add, cur_duration_list_addup, cur_duration_list) )
        cur_duration_list_addup =  list( map(lambda x: x * (1 / n_parts), cur_duration_list_addup) )
        cur_duration_list_addup.append(sum(cur_duration_list_addup))
        cognn_duration.append(cur_duration_list_addup)
    return cognn_duration

def plot_duration_breakdown_table(table_data):    
    # Create a DataFrame from the table_data list
    df = pd.DataFrame(table_data, columns=formatted_duration_tag_list)
    
    # Round each float number to two decimal places
    df = df.round(2)
    
    # Insert a column to the left of the DataFrame
    df.insert(0, 'Dataset', formatted_datasets)
    
    # Print the DataFrame as a markdown table
    print(df.to_markdown(index=False))

def get_comm(n_parts, executable_list, optimize_epoch_list, original_epoch_list, is_inference = False):
    cognn_comm = []

    preprocess_flag_list = ["Truepreprocess", "Falsepreprocess"]
    for executable in executable_list:
        for preprocess_flag in preprocess_flag_list:
            cur_exe_comm = []
            for dataset in datasets:
                cur_comm_file = cognn_comm_root_path + preprocess_flag + "_1scaler_" + executable + "_" + dataset + "_" + str(n_parts) + "p.comm"
                cur_comm = process_comm_file_and_get_veth_avg(cur_comm_file, 1, n_parts)
                cur_exe_comm.append(cur_comm)
            cognn_comm.append(cur_exe_comm)

    # optimize_epoch_list = [90, 90, 90]
    # original_epoch_list = [5, 2, 15]

    for i in range(len(datasets)):
        cognn_comm[0][i] = (cognn_comm[0][i] - cognn_comm[1][i]) / original_epoch_list[i]
        cognn_comm[2][i] = (cognn_comm[2][i] - cognn_comm[3][i]) / optimize_epoch_list[i]
        if not is_inference:
            cognn_comm[3][i] /= optimize_epoch_list[i]

    return cognn_comm

def plot_comm_table(table_data):    
    # Create a DataFrame from the table_data list
    df = pd.DataFrame(table_data, columns=["CoGNN Preprocess", "CoGNN Online", "CoGNN-Opt Preprocess", "CoGNN-Opt Online"])
    
    # Round each float number to two decimal places
    df = df.round(2)
    
    # Insert a column to the left of the DataFrame
    df.insert(0, 'Dataset', formatted_datasets)
    
    # Print the DataFrame as a markdown table
    print(df.to_markdown(index=False))

executable = "gcn-optimize"
optimize_duration = get_duration(2, [90, 90, 90], 90)
# print(optimize_duration)
for x in optimize_duration:
    print((x[0] + x[2] + x[4] + x[5]) / x[-1])
plot_duration_breakdown_table(optimize_duration)

executable = "gcn-original"
original_duration = get_duration(2, [5, 2, 15], 1)
for x in original_duration:
    print((x[0] + x[2] + x[4] + x[5]) / x[-1])
for i in range(len(original_duration)):
    print(original_duration[i][-1] / optimize_duration[i][-1])
# print(original_duration)
plot_duration_breakdown_table(original_duration)

comm_to_print = np.transpose(get_comm(2, ["gcn-original", "gcn-optimize"], [90, 90, 90], [5, 2, 15]))
plot_comm_table(comm_to_print)
for x in comm_to_print:
    print(x[0] + x[1], x[2] + x[3], (x[0] + x[1]) / (x[2] + x[3]))

# ------ Inference ------

cognn_log_root_path = "./../inference/log/"
cognn_comm_root_path = "./../inference/comm/"

executable = "gcn-inference-optimize"
optimize_duration = get_duration(2, [356, 372, 431], 1)
plot_duration_breakdown_table(optimize_duration)

executable = "gcn-original"
original_duration = get_duration(2, [5, 2, 15], 1)
plot_duration_breakdown_table(original_duration)

comm_to_print = np.transpose(get_comm(2, ["gcn-original", "gcn-inference-optimize"], [356, 372, 431], [5, 2, 15], is_inference=True))
plot_comm_table(comm_to_print)

