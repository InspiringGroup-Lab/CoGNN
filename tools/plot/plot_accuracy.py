import matplotlib.pyplot as plt
from operator import add, sub

from matplotlib import rcParams
import matplotlib as mpl
mpl.rcParams.update({'font.size': 11})

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
# datasets = ["cora", "citeseer"]
n_parts = 3
n_epochs = 90
cognn_log_file = []
cognn_test_acc = []
cognn_border_acc = []
fedgnn_test_acc = []
fedgnn_border_acc = []
plaingnn_test_acc = []
plaingnn_border_acc = []
fedgnn_log_root_path = "/work/Art/CoGNN/tools/mp-accuracy/log/"
plaingnn_log_root_path = "/work/Art/CoGNN/tools/mp-accuracy/log/"
figure_root_path = "/work/Art/CoGNN/tools/plot/figure/"
executable = "gcn-optimize"
cognn_log_root_path = "./../mp-accuracy/log/"


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

        
def plot_accuracies(cognn_test_acc, cognn_border_acc, fedgnn_test_acc, fedgnn_border_acc, plaingnn_test_acc, plaingnn_border_acc, save_path=None):
    diff_test_cognn_fedgnn = list( map(sub, cognn_test_acc, fedgnn_test_acc) )
    diff_border_cognn_fedgnn = list( map(sub, cognn_border_acc, fedgnn_border_acc) )
    print("diff test (cognn - fedgnn) of the last epoch: ", diff_test_cognn_fedgnn[-1])
    print("diff border (cognn - fedgnn) of the last epoch: ", diff_border_cognn_fedgnn[-1])

    diff_test_plaingnn_fedgnn = list( map(sub, plaingnn_test_acc, fedgnn_test_acc) )
    diff_border_plaingnn_fedgnn = list( map(sub, plaingnn_border_acc, fedgnn_border_acc) )
    print("diff test (plaingnn - fedgnn) of the last epoch: ", diff_test_plaingnn_fedgnn[-1])
    print("diff border (plaingnn - fedgnn) of the last epoch: ", diff_border_plaingnn_fedgnn[-1])

    # Create a figure and a set of subplots
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot test accuracies on the left axis
    axs[0].plot(cognn_test_acc, label='CoGNN Test Acc')
    axs[0].plot(fedgnn_test_acc, label='FedGNN Test Acc')
    axs[0].plot(plaingnn_test_acc, label='PlainGNN Test Acc')
    axs[0].set_title('Test Accuracy Comparison')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Accuracy (%)')
    axs[0].legend()
    
    # Plot border test accuracies on the right axis
    axs[1].plot(cognn_border_acc, label='CoGNN Border Test Acc')
    axs[1].plot(fedgnn_border_acc, label='FedGNN Border Test Acc')
    axs[1].plot(plaingnn_border_acc, label='PlainGNN Border Test Acc')
    axs[1].set_title('Border Test Accuracy Comparison')
    axs[1].set_xlabel('Epoch')
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
        
for i in range(len(datasets)):
    plot_accuracies(cognn_test_acc[i], cognn_border_acc[i], fedgnn_test_acc[i], fedgnn_border_acc[i], plaingnn_test_acc[i], plaingnn_border_acc[i], figure_root_path + datasets[i] + ".pdf")        



# Example usage:
# test_accuracy, border_accuracy = extract_accuracies('path_to_your_file.txt')
