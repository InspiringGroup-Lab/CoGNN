class GNNConfig:
    def __init__(self, num_layers, num_labels, input_dim, hidden_dim, num_samples, num_edges, learning_rate, train_ratio, val_ratio, test_ratio):
        self.num_layers = num_layers
        self.num_labels = num_labels
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_samples = num_samples
        self.num_edges = num_edges
        self.learning_rate = learning_rate
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

def read_gnn_config(file_path):
    config = {}
    with open(file_path, 'r') as file:
        for line in file:
            key, value = line.strip().split(' : ')
            config[key] = float(value) if '.' in value else int(value)
    
    return GNNConfig(**config)

def adjacency_matrix_comm(gnn_config: GNNConfig):
    sum_m = 2 * 8 * (gnn_config.input_dim + gnn_config.hidden_dim + 0 + gnn_config.hidden_dim)
    return sum_m * (gnn_config.num_samples**2 + gnn_config.num_samples) / (2 ** 30)

def adjacency_list_comm(gnn_config: GNNConfig):
    sum_m = 8 * (gnn_config.input_dim + gnn_config.hidden_dim + 0 + gnn_config.hidden_dim)
    return sum_m * (gnn_config.num_samples * gnn_config.num_edges) / (2 ** 30)

def graphsc_comm(gnn_config: GNNConfig):
    sum_m = 8 * (gnn_config.input_dim + gnn_config.hidden_dim + 0 + gnn_config.hidden_dim)
    return sum_m * 10 * (gnn_config.num_samples + gnn_config.num_edges) / (2 ** 30)

def cognn_comm(gnn_config: GNNConfig):
    sum_m = 8 * (gnn_config.input_dim + gnn_config.hidden_dim + 0 + gnn_config.hidden_dim)
    return sum_m * (gnn_config.num_samples + 5 * gnn_config.num_edges) / (2 ** 30)

def cognn_optimize_comm(gnn_config: GNNConfig):
    sum_m = 8 * (gnn_config.hidden_dim + gnn_config.num_labels + 0 + gnn_config.num_labels + gnn_config.hidden_dim + gnn_config.hidden_dim)
    return sum_m * (gnn_config.num_samples + 5 * gnn_config.num_edges) / (2 ** 30)

datasets = ["cora", "citeseer", "pubmed"]
config_root_path = "./../config/"

# Define a function to print the markdown table
def print_comm_table(gnn_configs):
    # Define the headers of the table
    headers = ["Dataset", "Adjacency Matrix", "Adjacency List", "GraphSC", "CoGNN", "CoGNN Optimized"]
    
    # Start the markdown table with headers
    markdown_table = "| " + " | ".join(headers) + " |\n"
    markdown_table += "|---" * len(headers) + "|\n"
    
    # Iterate over each dataset and GNNConfig object to compute communications
    for dataset, gnn_config in zip(datasets, gnn_configs):
        # Compute communications for each method
        adj_matrix_comm = "{:.2f}".format(adjacency_matrix_comm(gnn_config))
        adj_list_comm = "{:.2f}".format(adjacency_list_comm(gnn_config))
        graphsc_comm_val = "{:.2f}".format(graphsc_comm(gnn_config))
        cognn_comm_val = "{:.2f}".format(cognn_comm(gnn_config))
        cognn_opt_comm_val = "{:.2f}".format(cognn_optimize_comm(gnn_config))

        print(graphsc_comm(gnn_config) / cognn_comm(gnn_config), graphsc_comm(gnn_config) / cognn_optimize_comm(gnn_config))
        
        # Add a row to the markdown table for the current dataset
        row = f"| {dataset} | {adj_matrix_comm} | {adj_list_comm} | {graphsc_comm_val} | {cognn_comm_val} | {cognn_opt_comm_val} |"
        markdown_table += row + "\n"
    
    # Print the markdown table
    print(markdown_table)

# Read GNN configurations for each dataset
gnn_configs = [read_gnn_config(config_root_path + dataset + "_config.txt") for dataset in datasets]

# Print the communication comparison table
print_comm_table(gnn_configs)


