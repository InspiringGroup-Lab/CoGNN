num_inter_edges = [[5404, 7184, 8028, 8552], \
                    [4668, 6040, 6996, 7354], \
                    [44278, 59248, 66424, 71202]]

num_edges = [10556, 9104, 88648]

list_num_parts = [2, 3, 4, 5]

for i in range(len(num_edges)):
    for j in range(len(list_num_parts)):
        print(f"{num_inter_edges[i][j] / num_edges[i]:.2f}")