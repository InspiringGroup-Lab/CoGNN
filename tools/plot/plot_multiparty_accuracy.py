import matplotlib.pyplot as plt
from operator import add, sub
from matplotlib.ticker import FormatStrFormatter
import pandas as pd
from matplotlib import rcParams
import matplotlib as mpl
mpl.rcParams.update({'font.size': 11})

markerList = ['^', 'o', 'x', '*']

colors = ["teal", "salmon", "skyblue"]

def extract_cognn_accuracies(file_path):
    # Arrays to store the accuracies
    test_set_accuracy = []
    border_test_set_accuracy = []
    
    # Open the file and read line by line
    with open(file_path, 'r') as file:
        for line in file:
            # Check if the line contains 'border test set accuracy'
            if 'border test set accuracy' in line:
                # Extract and store the border test set accuracy
                accuracy_value = float(line.split('=')[1].strip())
                border_test_set_accuracy.append(accuracy_value)
            # Check if the line contains 'test set accuracy'
            elif 'test set accuracy' in line:
                # Extract and store the test set accuracy
                accuracy_value = float(line.split('=')[1].strip())
                test_set_accuracy.append(accuracy_value)
    
    return test_set_accuracy, border_test_set_accuracy

def extract_plain_accuracies(file_path, part = -1):
    # Arrays to store the accuracies
    test_accuracy = []
    test_is_border_accuracy = []
    
    if part < 0:
        # Open the file and read line by line
        with open(file_path, 'r') as file:
            for line in file:
                # Check if the line contains 'Test accuracy'
                if 'Test accuracy' in line:
                    # Extract and store the Test accuracy
                    accuracy_value = float(line.split('Test accuracy:')[1].split('%')[0].strip())
                    test_accuracy.append(accuracy_value)
                    
                # Check if the line contains 'Test is_border accuracy'
                if 'Test is_border accuracy' in line:
                    # Extract and store the Test is_border accuracy
                    accuracy_value = float(line.split('Test is_border accuracy:')[1].split('%')[0].strip())
                    test_is_border_accuracy.append(accuracy_value)
    else:
        # Open the file and read line by line
        with open(file_path, 'r') as file:
            for line in file:
                if 'Part: ' + str(part) in line:
                    # Check if the line contains 'Test accuracy'
                    if 'Test accuracy' in line:
                        # Extract and store the Test accuracy
                        accuracy_value = float(line.split('Test accuracy:')[1].split('%')[0].strip())
                        test_accuracy.append(accuracy_value)
                        
                    # Check if the line contains 'Test is_border accuracy'
                    if 'Test is_border accuracy' in line:
                        # Extract and store the Test is_border accuracy
                        accuracy_value = float(line.split('Test is_border accuracy:')[1].split('%')[0].strip())
                        test_is_border_accuracy.append(accuracy_value)                
    
    return test_accuracy, test_is_border_accuracy

datasets = ["cora", "citeseer", "pubmed"]
formatted_datasets = ["Cora", "CiteSeer", "PubMed"]
# datasets = ["cora", "citeseer"]
n_parts_list = [2, 3, 4, 5]
n_epochs = 90
cognn_log_file = []
fedgnn_log_root_path = "/home/zzh/project/test-GCN/Art/build/bin/mp-accuracy/"
plaingnn_log_root_path = "/home/zzh/project/test-GCN/Art/build/bin/mp-accuracy/"
figure_root_path = "/home/zzh/project/test-GCN/Art/CoGNN/tools/plot/figure/multi-party/"
executable = "gcn-optimize"
cognn_log_root_path = "./../mp-accuracy/log/"

def get_acc(n_parts):
    cognn_test_acc = []
    cognn_border_acc = []
    fedgnn_test_acc = []
    fedgnn_border_acc = []
    plaingnn_test_acc = []
    plaingnn_border_acc = []
    for dataset in datasets:
        cur_test_acc_addup = [0] * n_epochs
        cur_border_acc_addup = [0] * n_epochs
        for i in range(n_parts):
            cur_log_file = cognn_log_root_path + executable + "/" + dataset + "/" + str(n_parts) + "p/" + "gcn_test_" + dataset + "_" + str(i) + ".log"
            test_accuracy, border_accuracy = extract_cognn_accuracies(cur_log_file)
            test_accuracy = test_accuracy[:n_epochs]
            border_accuracy = border_accuracy[:n_epochs]
            cur_test_acc_addup = list( map(add, cur_test_acc_addup, test_accuracy) )
            cur_border_acc_addup = list( map(add, cur_border_acc_addup, border_accuracy) )
        cur_test_acc_addup =  list( map(lambda x: x * (1 / n_parts), cur_test_acc_addup) )
        cur_border_acc_addup =  list( map(lambda x: x * (1 / n_parts), cur_border_acc_addup) )
        cognn_test_acc.append(cur_test_acc_addup)
        cognn_border_acc.append(cur_border_acc_addup)    

        cur_test_acc_addup = [0] * n_epochs
        cur_border_acc_addup = [0] * n_epochs
        for i in range(n_parts):
            cur_log_file = fedgnn_log_root_path + "fed-gcn." + dataset + "." + str(n_parts) + "p.log"
            test_accuracy, border_accuracy = extract_plain_accuracies(cur_log_file, i)
            test_accuracy = test_accuracy[:n_epochs]
            border_accuracy = border_accuracy[:n_epochs]
            cur_test_acc_addup = list( map(add, cur_test_acc_addup, test_accuracy) )
            cur_border_acc_addup = list( map(add, cur_border_acc_addup, border_accuracy) )
        cur_test_acc_addup =  list( map(lambda x: x * (1 / n_parts), cur_test_acc_addup) )
        cur_border_acc_addup =  list( map(lambda x: x * (1 / n_parts), cur_border_acc_addup) )
        fedgnn_test_acc.append(cur_test_acc_addup)
        fedgnn_border_acc.append(cur_border_acc_addup)    

        cur_log_file = plaingnn_log_root_path + "plaintext-gcn." + dataset + ".log"
        test_accuracy, border_accuracy = extract_plain_accuracies(cur_log_file)
        test_accuracy = test_accuracy[:n_epochs]
        border_accuracy = border_accuracy[:n_epochs]
        plaingnn_test_acc.append(test_accuracy)
        plaingnn_border_acc.append(border_accuracy)
    return cognn_test_acc, cognn_border_acc, fedgnn_test_acc, fedgnn_border_acc, plaingnn_test_acc, plaingnn_border_acc

# cur_l, = ax.plot(fig_data[executableList[i]]["x"], fig_data[executableList[i]][str(variable)], 'o', ls='-', ms=4, label=str(variable))
        
def plot_multiparty_accuracies_for_each_dataset(cognn_test_acc, cognn_border_acc, fedgnn_test_acc, fedgnn_border_acc, plaingnn_test_acc, plaingnn_border_acc, save_path=None):
    # Create a figure and a set of subplots
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot test accuracies on the left axis
    axs[0].plot([str(x) for x in n_parts_list], cognn_test_acc, markerList[0], ls='-', ms=4, label='CoGNN')
    axs[0].plot([str(x) for x in n_parts_list], fedgnn_test_acc, markerList[1], ls='-', ms=4, label='FedGNN')
    axs[0].plot([str(x) for x in n_parts_list], plaingnn_test_acc, markerList[2], ls='-', ms=4, label='PlainGNN')
    axs[0].set_title('Test Accuracy Comparison')
    axs[0].set_xlabel('Number of Parties')
    axs[0].set_ylabel('Accuracy (%)')
    axs[0].legend()
    
    # Plot border test accuracies on the right axis
    axs[1].plot([str(x) for x in n_parts_list], cognn_border_acc, markerList[0], ls='-', ms=4, label='CoGNN Border')
    axs[1].plot([str(x) for x in n_parts_list], fedgnn_border_acc, markerList[1], ls='-', ms=4, label='FedGNN Border')
    axs[1].plot([str(x) for x in n_parts_list], plaingnn_border_acc, markerList[2], ls='-', ms=4, label='PlainGNN Border')
    axs[1].set_title('Border Test Accuracy Comparison')
    axs[1].set_xlabel('Number of Parties')
    axs[1].set_ylabel('Accuracy (%)')
    axs[1].legend()
    
    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Save the figure if a save path is provided
    if save_path:
        plt.savefig(save_path)
        print(f"Figure saved as {save_path}")
    
    # # Show the plot
    # plt.show()
        
# for i in range(len(datasets)):
#     plot_accuracies(cognn_test_acc[i], cognn_border_acc[i], fedgnn_test_acc[i], fedgnn_border_acc[i], plaingnn_test_acc[i], plaingnn_border_acc[i], figure_root_path + datasets[i] + ".pdf")        

# Define a function to plot the accuracies for a given dataset
def subplot_accuracies_for_dataset(axs, dataset, index, cognn_test_acc, cognn_border_acc, fedgnn_test_acc, fedgnn_border_acc, plaingnn_test_acc, plaingnn_border_acc):
    # Get the row and column index of the subplot
    row = index
    col = 0

    if row == 0:
        axs[row, col].set_ylim([60, 95])
        axs[row, col+1].set_ylim([60, 95])
    elif row == 1:
        axs[row, col].set_ylim([20, 90])
        axs[row, col+1].set_ylim([20, 90])
    elif row == 2:
        axs[row, col].set_ylim([75, 90])
        axs[row, col+1].set_ylim([75, 90])

    axs[row, col].set_xlim([-0.3, 3.3])
    axs[row, col+1].set_xlim([-0.3, 3.3])
    
    # Plot test accuracies on the left axis
    axs[row, col].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    axs[row, col].plot([str(x) for x in n_parts_list], cognn_test_acc, markerList[0], linewidth=3, ls='-', ms=8, color=colors[0], label='CoGNN')
    axs[row, col].plot([str(x) for x in n_parts_list], fedgnn_test_acc, markerList[1], linewidth=3, ls='-', ms=8, color=colors[1], label='FedGNN')
    axs[row, col].plot([str(x) for x in n_parts_list], plaingnn_test_acc, markerList[2], linewidth=3, ls='-', ms=8, color=colors[2], label='PlainGNN')
    for i in range(len(n_parts_list)):
        label = "+{:.2f}".format(cognn_test_acc[i] - fedgnn_test_acc[i])
        axs[row, col].annotate(label, # this is the text
                    (str(n_parts_list[i]), cognn_test_acc[i]), # these are the coordinates to position the label
                    textcoords="offset points", # how to position the text
                    xytext=(0,10), # distance from text to points (x,y)
                    ha='center',
                    color=colors[0]) # horizontal alignment can be left, right or center
    axs[row, col].set_title(f'Accuracy for {dataset}')
    axs[row, col].set_xlabel('Number of Parties')
    axs[row, col].set_ylabel('Accuracy (%)')
    axs[row, col].legend()
    
    # Plot border test accuracies on the right axis
    axs[row, col+1].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    axs[row, col+1].plot([str(x) for x in n_parts_list], cognn_border_acc, markerList[0], linewidth=3, ls='-', ms=8, color=colors[0], label='CoGNN')
    axs[row, col+1].plot([str(x) for x in n_parts_list], fedgnn_border_acc, markerList[1], linewidth=3, ls='-', ms=8, color=colors[1], label='FedGNN')
    axs[row, col+1].plot([str(x) for x in n_parts_list], plaingnn_border_acc, markerList[2], linewidth=3, ls='-', ms=8, color=colors[2], label='PlainGNN')
    for i in range(len(n_parts_list)):
        label = "+{:.2f}".format(cognn_border_acc[i] - fedgnn_border_acc[i])
        axs[row, col + 1].annotate(label, # this is the text
                    (str(n_parts_list[i]), cognn_border_acc[i]), # these are the coordinates to position the label
                    textcoords="offset points", # how to position the text
                    xytext=(0,10), # distance from text to points (x,y)
                    ha='center',
                    color=colors[0]) # horizontal alignment can be left, right or center
    axs[row, col+1].set_title(f'Border Accuracy for {dataset}')
    axs[row, col+1].set_xlabel('Number of Parties')
    axs[row, col+1].set_ylabel('Accuracy (%)')
    axs[row, col+1].legend()


# Example usage:
# test_accuracy, border_accuracy = extract_accuracies('path_to_your_file.txt')
        
cognn_test_acc = [ [] for i in range(len(datasets))]
cognn_border_acc = [ [] for i in range(len(datasets))]
fedgnn_test_acc = [ [] for i in range(len(datasets))]
fedgnn_border_acc = [ [] for i in range(len(datasets))]
plaingnn_test_acc = [ [] for i in range(len(datasets))]
plaingnn_border_acc = [ [] for i in range(len(datasets))]

part_to_acc = {}
for n_parts in n_parts_list:
    part_to_acc[n_parts] = get_acc(n_parts)
    for i in range(len(datasets)):
        # print(i, part_to_acc[n_parts][0][i][-1])
        cognn_test_acc[i].append(part_to_acc[n_parts][0][i][-1])
        # print(cognn_test_acc)
        cognn_border_acc[i].append(part_to_acc[n_parts][1][i][-1])
        fedgnn_test_acc[i].append(part_to_acc[n_parts][2][i][-1])
        fedgnn_border_acc[i].append(part_to_acc[n_parts][3][i][-1])
        plaingnn_test_acc[i].append(part_to_acc[n_parts][4][i][-1])
        plaingnn_border_acc[i].append(part_to_acc[n_parts][5][i][-1])

# for i in range(len(datasets)):
#     plot_multiparty_accuracies_for_each_dataset(cognn_test_acc[i], cognn_border_acc[i], fedgnn_test_acc[i], fedgnn_border_acc[i], plaingnn_test_acc[i], plaingnn_border_acc[i], figure_root_path + datasets[i] + ".pdf")  

def plot_multiparty_accuracies():
    # Create a figure and a set of subplots
    fig, axs = plt.subplots(3, 2, figsize=(8, 8))

    # Loop through the datasets and plot the accuracies for each one
    for i in range(len(datasets)):
        subplot_accuracies_for_dataset(axs, formatted_datasets[i], i, cognn_test_acc[i], cognn_border_acc[i], fedgnn_test_acc[i], fedgnn_border_acc[i], plaingnn_test_acc[i], plaingnn_border_acc[i])
        print(fedgnn_test_acc[i][0] - fedgnn_test_acc[i][-1])
        print(fedgnn_border_acc[i][0] - fedgnn_border_acc[i][-1])        
        print(cognn_test_acc[i][-1] - fedgnn_test_acc[i][-1])
        print(cognn_border_acc[i][-1] - fedgnn_border_acc[i][-1])
        print("---")

    plt.subplots_adjust(left=0.08, bottom=0.06, right=0.99, top=0.97, wspace=0.2, hspace=0.4)

    save_path = figure_root_path + "multiparty_accuracy.pdf"
    # Save the figure if a save path is provided
    if save_path:
        plt.savefig(save_path)
        print(f"Figure saved as {save_path}")

def plot_accuracy_table(index):
    table_data = []
    for i in range(len(datasets)):
        table_data.append([plaingnn_test_acc[i][index], plaingnn_border_acc[i][index], cognn_test_acc[i][index], cognn_border_acc[i][index], fedgnn_test_acc[i][index], fedgnn_border_acc[i][index]])
    
    # Create a DataFrame from the table_data list
    df = pd.DataFrame(table_data, columns=["Plain GNN", "Plain GNN Border", "CoGNN", "CoGNN Border", "FedGNN", "FedGNN Border"])
    
    # Round each float number to two decimal places
    df = df.round(2)
    
    # Insert a column to the left of the DataFrame
    df.insert(0, 'Dataset', formatted_datasets)
    
    # Print the DataFrame as a markdown table
    print(df.to_markdown(index=False))

plot_multiparty_accuracies()
plot_accuracy_table(0)
plot_accuracy_table(-1)