import numpy as np
from torch_geometric.datasets import Planetoid
import os

def my_makedir(path):
    if not os.path.isdir(path):
        os.makedirs(path)

list_partitions = [2, 3, 4, 5]

def is_undirected (edge_indices):
  # Convert the edge_indices to a set of tuples
  edge_set = set (map (tuple, edge_indices.T))

  # Check if every edge has a reverse edge
  return all ((v, u) in edge_set for (u, v) in edge_set)

# Define the function to output a vertex partition file
def output_vertex_partition(vertices, partitions, output_file):
    # Open the output file in write mode
    with open(output_file, "w") as f:
        # Loop through each vertex
        for vertex in range(vertices):
            # Calculate the partition number for the vertex
            partition = vertex % partitions
            # Write the vertex and the partition number separated by a tab
            f.write(f"{vertex}\t{partition}\n")

def transform_dataset(data_name, data_dir):
  # Load the Cora dataset
  dataset = Planetoid (root=data_dir, name=data_name)
  dataset = dataset.shuffle()
  data = dataset [0]

  # Extract the node features, node labels, and edge indices
  node_features = data.x.numpy ()
  node_labels = data.y.numpy ()
  edge_indices = data.edge_index.numpy ()

  print(is_undirected(edge_indices))

  # Add a column of node indices to the left of node features
  node_indices = np.arange (data.num_nodes) [:, None]
  node_features = np.concatenate ((node_indices, node_features), axis=1)

  # Concatenate the node features and node labels
  vertex_list = np.concatenate ((node_features, node_labels [:, None]), axis=1)

  transformed_path = data_dir + data_name + "/transformed/"
  my_makedir(transformed_path)

  # Save the vertex list to a txt file
  # print(data.num_features)
  # print(vertex_list.shape[1])
  fmt = '%d ' + '%f ' * (data.num_features) + '%d'
  np.savetxt (transformed_path + data_name.lower() + '.vertex.preprocessed', vertex_list, fmt=fmt)

  # Transpose the edge indices
  edge_list = edge_indices.T

  # Save the edge list to a txt file
  np.savetxt (transformed_path + data_name.lower() + '.edge.preprocessed', edge_list, fmt='%d')

  for p in list_partitions:
      output_vertex_partition(data.num_nodes, p, transformed_path + data_name.lower() + '.part.preprocessed.' + str(p) + "p")

def transform_dataset_by_parts(data_name, data_dir, num_parts):
  # Load the Cora dataset
  dataset = Planetoid (root=data_dir, name=data_name)
  dataset = dataset.shuffle()
  data = dataset [0]

  # Extract the node features, node labels, and edge indices
  node_features = data.x.numpy ()
  node_labels = data.y.numpy ()
  edge_indices = data.edge_index.numpy ()

  print(is_undirected(edge_indices))

  # Add a column of node indices to the left of node features
  node_indices = np.arange (data.num_nodes) [:, None]
  node_features = np.concatenate ((node_indices, node_features), axis=1)

  # Concatenate the node features and node labels
  vertex_list = np.concatenate ((node_features, node_labels [:, None]), axis=1)

  transformed_path = data_dir + data_name + "/transformed/"
  my_makedir(transformed_path)

  # Split the vertex list into num_parts sub-lists
  vertex_sublists = np.array_split(vertex_list, num_parts)

  edge_list = edge_indices.T

  # For each sub-list, combine it with the previous ones and save as a new dataset
  for i in range(1, num_parts):
    combined_vertex_list = np.concatenate(vertex_sublists[:i+1], axis=0)
    print(combined_vertex_list.shape[0])
    # print(vertex_sublists[i].shape[0])
    # Save the combined vertex list to a txt file
    fmt = '%d ' + '%f ' * (data.num_features) + '%d'
    sub_path = transformed_path + str(i + 1) + "s/"
    my_makedir(sub_path)
    np.savetxt(sub_path + data_name.lower() + f'.vertex.preprocessed', combined_vertex_list, fmt=fmt)

    # Get the node indices of the current sub-dataset
    sub_node_indices = combined_vertex_list[:, 0]
    # Filter the edge list by checking if both nodes are in the node indices
    filtered_edge_list = edge_list[np.isin(edge_list, sub_node_indices).all(axis=1)]
    print(filtered_edge_list.shape[0])
    # Save the filtered edge list to a txt file
    np.savetxt(sub_path + data_name.lower() + f'.edge.preprocessed', filtered_edge_list, fmt='%d')

    sub_part_labels = np.repeat(0, vertex_sublists[0].shape[0])[:,None]
    for k in range(1, i+1):
      sub_part_labels = np.concatenate((sub_part_labels, np.repeat(k, vertex_sublists[k].shape[0])[:,None]), axis = 0)
    sub_part_labels = np.concatenate((combined_vertex_list[:, 0][:,None], sub_part_labels), axis = 1)
    np.savetxt(sub_path + data_name.lower() + f'.part.preprocessed', sub_part_labels, fmt='%d')

  # # Save the vertex list to a txt file
  # # print(data.num_features)
  # # print(vertex_list.shape[1])
  # fmt = '%d ' + '%f ' * (data.num_features) + '%d'
  # np.savetxt (transformed_path + data_name.lower() + '.vertex.preprocessed', vertex_list, fmt=fmt)

  # # Transpose the edge indices
  # edge_list = edge_indices.T

  # # Save the edge list to a txt file
  # np.savetxt (transformed_path + data_name.lower() + '.edge.preprocessed', edge_list, fmt='%d')

  # for p in list_partitions:
  #     output_vertex_partition(data.num_nodes, p, transformed_path + data_name.lower() + '.part.preprocessed.' + str(p) + "p")

# transform_dataset("Pubmed", "./data/")
# transform_dataset("Citeseer", "./data/")
# transform_dataset("Cora", "./data/")

transform_dataset_by_parts("Pubmed", "./data/", 5)
transform_dataset_by_parts("Citeseer", "./data/", 5)