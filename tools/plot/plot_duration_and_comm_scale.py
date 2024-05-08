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

def extract_durations(file_path, match_lines):
    # print(file_path)
    # Arrays to store the durations
    iter_duration = []

    # Open the file and read line by line
    with open(file_path, 'r') as file:
        lines = file.readlines()
        cnt = 0
        for index, line in enumerate(lines):
            if '::iteration took' in line:
                # Extract and store the border test set accuracy
                # print(line)
                cnt+=1
                try:
                    cur_duration = float(line.split(' ')[2].strip())
                except:
                    cur_index = index
                    while True:
                        if 'seconds' in lines[cur_index]:
                            cur_duration = float(lines[cur_index].split(" seconds")[0].split(' ')[-1])
                            break
                        cur_index += 1
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

log_root_path = "./../log/"
config_root_path = "./../config/"
comm_root_path = "./../comm/"

eval_setting_table = [ \
    ["graphsc", "test-graphsc"], \
    ["cognn-scale", "gcn-original"], \
    ["cognn-scale", "gcn-optimize"] \
]

colors = ["salmon", "skyblue", "teal"]

formatted_eval_settings = ["GraphSC", "CoGNN", "CoGNN-Opt"]

def set_root_paths(application):
    global log_root_path, config_root_path, preprocess_root_path, comm_root_path
    log_root_path = f"./../{str(application)}/log/"
    config_root_path = "./../config/"
    comm_root_path = f"./../{str(application)}/comm/"

def set_num_iters_per_epoch(executable):
    global num_iter_per_epoch
    if executable == "gcn-original" or executable == "test-graphsc":
        num_iter_per_epoch = 4
    elif executable == "gcn-optimize":
        num_iter_per_epoch = 6

def get_duration(application, executable, dataset, scale, n_parties):
    set_root_paths(application)
    set_num_iters_per_epoch(executable)
    duration = []
    cur_duration_addup = [0] * avg_epochs
    for i in range(n_parties):
        cur_log_file = log_root_path + executable + "/" + dataset + "/" + str(scale) + "s/" + "noPreprocess/" + "gcn_test_" + dataset + "_" + str(i) + ".log"
        cur_iter_duration = extract_durations(cur_log_file, avg_epochs * num_iter_per_epoch)
        cur_epoch_duration = sum_consecutive_durations(cur_iter_duration, num_iter_per_epoch)
        cur_epoch_duration = cur_epoch_duration[:avg_epochs]
        cur_duration_addup = list( map(add, cur_duration_addup, cur_epoch_duration) )
    cur_duration_addup =  list( map(lambda x: x * (1 / n_parties), cur_duration_addup) )
    duration.append(sum(cur_duration_addup) / avg_epochs)
    return duration

def get_comm(application, executable, dataset, scale, n_parties):
    set_root_paths(application)
    cognn_comm = []
    cur_comm_file = comm_root_path + "Falsepreprocess" + "_1scaler_" + executable + "_" + dataset + "_" + str(scale)
    if application == "graphsc":
        cur_comm_file += "s.comm"
    else:
        cur_comm_file += "p.comm"
    cur_comm = process_comm_file_and_get_veth_avg(cur_comm_file, avg_epochs, n_parties)
    cognn_comm.append(cur_comm)
    return cognn_comm

def plot_duration_and_comm(axs, duration, comm, dataset_id):

    # axs[0].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    axs[0].yaxis.set_major_formatter(FormatStrFormatter('%d'))
    for i in range(len(eval_setting_table)):
        axs[0].plot([str(x) for x in n_parts_list], duration[i], markerList[i], linewidth=3, ls='-', ms=8, color=colors[i], label=formatted_eval_settings[i])
        # print(cognn_duration[i][0], cognn_duration[i][-1], cognn_duration[i][0] / cognn_duration[i][-1])
    axs[0].set_title(f'Duration for {formatted_datasets[dataset_id]}')
    axs[0].set_xlabel('Number of Graph Owners')
    axs[0].set_ylabel('Duration per Epoch [s]')
    axs[0].legend()
    axs[0].yaxis.set_tick_params(labelbottom=True, rotation=45)
    
    # axs[1].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    axs[1].yaxis.set_major_formatter(FormatStrFormatter('%d'))    
    for i in range(len(eval_setting_table)):
        axs[1].plot([str(x) for x in n_parts_list], comm[i], markerList[i], linewidth=3, ls='-', ms=8, color=colors[i], label=formatted_eval_settings[i])
        # print(cognn_comm[i][0], cognn_comm[i][-1], cognn_comm[i][0] / cognn_comm[i][-1])
    axs[1].set_title(f'Communication for {formatted_datasets[dataset_id]}')
    axs[1].set_xlabel('Number of Graph Owners')
    axs[1].set_ylabel('Comm per Party [GB]')
    axs[1].legend()
    axs[1].yaxis.set_tick_params(labelbottom=True, rotation=45)


    
    # # Show the plot
    # plt.show()

durations = [] # For different datasets
for dataset in datasets:
    duration = []
    for line in eval_setting_table:
        cur_duration = []
        for n_parts in n_parts_list:
            if line[0] == "graphsc":
                cur_duration.append(get_duration(line[0], line[1], dataset, n_parts, 2))
            else:
                cur_duration.append(get_duration(line[0], line[1], dataset, n_parts, n_parts))
        duration.append(np.transpose(cur_duration)[0])
    duration = np.array(duration)
    # duration[0] /= 2 * np.array(range(1, 5))
    duration[0] /= 2
    print(duration)
    print("Duration GraphSC vs CoGNN", duration[0] / duration[1])
    print("Duration GraphSC vs CoGNN-Opt", duration[0] / duration[2])
    print("Duration CoGNN vs CoGNN-Opt", duration[1] / duration[2])
    print("Duration Growth", np.transpose(duration)[-1] / np.transpose(duration)[0])
    durations.append(duration)

comms = []
for dataset in datasets:
    comm = []
    for line in eval_setting_table:
        cur_comm = []
        for n_parts in n_parts_list:
            if line[0] == "graphsc":
                cur_comm.append(get_comm(line[0], line[1], dataset, n_parts, 2))
            else:
                cur_comm.append(get_comm(line[0], line[1], dataset, n_parts, n_parts))
        comm.append(np.transpose(cur_comm)[0])
    comm = np.array(comm)
    print(comm)
    print("Comm GraphSC vs CoGNN", comm[0] / comm[1])
    print("Comm GraphSC vs CoGNN-Opt", comm[0] / comm[2])
    print("Comm CoGNN vs CoGNN-Opt", comm[1] / comm[2])
    print("Comm Growth", np.transpose(comm)[-1] / np.transpose(comm)[0])
    comms.append(comm)

# comm = []
# for n_parts in n_parts_list:
#     comm.append(get_comm("graphsc", "test-graphsc", n_parts, 2))
# comm = np.transpose(comm)
# print(comm)

# plot_duration_and_comm(duration, comm, figure_root_path + "duration_comm_scale.pdf")

save_path = figure_root_path + "duration_comm_scale.pdf"

# Create a figure and a set of subplots
fig, axs = plt.subplots(3, 2, figsize=(8, 8))

for i in range(len(datasets)):
    plot_duration_and_comm(axs[i], durations[i], comms[i], i)

plt.subplots_adjust(left=0.092, bottom=0.06, right=0.99, top=0.97, wspace=0.2, hspace=0.4)

# Save the figure if a save path is provided
if save_path:
    plt.savefig(save_path)
    print(f"Figure saved as {save_path}")
