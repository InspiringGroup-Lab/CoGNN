import matplotlib.pyplot as plt
from operator import add, sub
from matplotlib.ticker import FormatStrFormatter
import numpy as np

from matplotlib import rcParams
import matplotlib as mpl
mpl.rcParams.update({'font.size': 11})

markerList = ['^', 'o', 'x', '*']

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

def extract_cognn_durations(file_path, match_lines):
    # Arrays to store the durations
    iter_duration = []
    
    # Open the file and read line by line
    with open(file_path, 'r') as file:
        cnt = 0
        for line in file:
            if '::iteration took' in line:
                # Extract and store the border test set accuracy
                # print(line)
                cnt+=1
                cur_duration = float(line.split(' ')[2].strip())
                iter_duration.append(cur_duration)
                if cnt >= match_lines:
                    break
    
    return iter_duration

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
n_parts_list = [2, 3, 4, 5]
n_epochs = 90
avg_epochs = 1
num_iter_per_epoch = 6
cognn_log_file = []
figure_root_path = "/work/Art/CoGNN/tools/plot/figure/multi-party/"
executable = "gcn-optimize"
cognn_log_root_path = "./../log/"
cognn_comm_root_path = "./../comm/"
preprocess_flag = "Truepreprocess"

def get_duration(n_parts):
    cognn_duration = []
    for dataset in datasets:
        cur_duration_addup = [0] * avg_epochs
        for i in range(n_parts):
            cur_log_file = cognn_log_root_path + executable + "/" + dataset + "/" + str(n_parts) + "p/" + "gcn_test_" + dataset + "_" + str(i) + ".log"
            cur_iter_duration = extract_cognn_durations(cur_log_file, avg_epochs * num_iter_per_epoch)
            cur_epoch_duration = sum_consecutive_durations(cur_iter_duration, num_iter_per_epoch)
            cur_epoch_duration = cur_epoch_duration[:avg_epochs]
            cur_duration_addup = list( map(add, cur_duration_addup, cur_epoch_duration) )
        cur_duration_addup =  list( map(lambda x: x * (1 / n_parts), cur_duration_addup) )
        cognn_duration.append(sum(cur_duration_addup) / avg_epochs)
    return cognn_duration

def get_comm(n_parts):
    cognn_comm = []
    for dataset in datasets:
        cur_comm_file = cognn_comm_root_path + preprocess_flag + "_1scaler_" + executable + "_" + dataset + "_" + str(n_parts) + "p.comm"
        cur_comm = process_comm_file_and_get_veth_avg(cur_comm_file, n_epochs, n_parts)
        cognn_comm.append(cur_comm)
    return cognn_comm

def plot_duration_and_comm(cognn_duration, cognn_comm, save_path=None):
    # Create a figure and a set of subplots
    fig, axs = plt.subplots(1, 2, figsize=(8, 2.5))
    
    axs[0].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    # axs[0].xaxis.set_major_formatter(FormatStrFormatter('%d'))
    for i in range(len(formatted_datasets)):
        axs[0].plot([str(x) for x in n_parts_list], cognn_duration[i], markerList[i], linewidth=2, ls='-', ms=6, label=formatted_datasets[i])
        print(cognn_duration[i][0], cognn_duration[i][-1], cognn_duration[i][0] / cognn_duration[i][-1])
    axs[0].set_xlabel('Number of Parties')
    axs[0].set_ylabel('Duration per Epoch [s]')
    axs[0].legend()
    
    axs[1].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    # axs[1].xaxis.set_major_formatter(FormatStrFormatter('%d'))    
    for i in range(len(formatted_datasets)):
        axs[1].plot([str(x) for x in n_parts_list], cognn_comm[i], markerList[i], linewidth=2, ls='-', ms=6, label=formatted_datasets[i])
        print(cognn_comm[i][0], cognn_comm[i][-1], cognn_comm[i][0] / cognn_comm[i][-1])
    axs[1].set_xlabel('Number of Parties')
    axs[1].set_ylabel('Comm per Epoch [GB]')
    axs[1].legend()

    plt.subplots_adjust(left=0.095, bottom=0.17, right=0.99, top=0.97, wspace=0.21, hspace=0.4)

    # Save the figure if a save path is provided
    if save_path:
        plt.savefig(save_path)
        print(f"Figure saved as {save_path}")
    
    # # Show the plot
    # plt.show()

cognn_duration = []
for n_parts in n_parts_list:
    cognn_duration.append(get_duration(n_parts))
cognn_duration = np.transpose(cognn_duration)
print(cognn_duration)

cognn_comm = []
for n_parts in n_parts_list:
    cognn_comm.append(get_comm(n_parts))
cognn_comm = np.transpose(cognn_comm)
print(cognn_comm)

plot_duration_and_comm(cognn_duration, cognn_comm, figure_root_path + "duration_comm.pdf")


