
from tqdm import tqdm
import numpy as np
import itertools
import pandas as pd
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def build_edge_info(nodes_type_A,nodes_type_C,graph):
    if not nodes_type_A or not nodes_type_C:
        raise ValueError("Error: nodes_type_A or nodes_type_C is empty. Cannot proceed.")

    src, dst = graph.edges()
    edge_info_matrix = torch.zeros((len(nodes_type_A), len(nodes_type_C)), device=device)
    node_to_type_A_index = {node_id: idx for idx, node_id in enumerate(nodes_type_A)}
    node_to_type_C_index = {node_id: idx for idx, node_id in enumerate(nodes_type_C)}
    for u, v in zip(src.tolist(), dst.tolist()):
        if u in node_to_type_A_index and v in node_to_type_C_index:
            a_idx = node_to_type_A_index[u]
            c_idx = node_to_type_C_index[v]
            edge_info_matrix[a_idx, c_idx] = 1.0
    edge_info_df = pd.DataFrame(edge_info_matrix.cpu().numpy(), index=nodes_type_A, columns=nodes_type_C)
    return edge_info_df

def Hadamard_product(embeddings_A, embeddings_C, nodes_type_A, nodes_type_C, batch_size=1000):

    if isinstance(embeddings_A, np.ndarray):
        embeddings_A = torch.tensor(embeddings_A, device=device)
    else:
        embeddings_A = embeddings_A.clone().detach().to(device)
    if isinstance(embeddings_C, np.ndarray):
        embeddings_C = torch.tensor(embeddings_C, device=device)
    else:
        embeddings_C = embeddings_C.clone().detach().to(device)

    combinations = list(itertools.product(nodes_type_A, nodes_type_C))
    print(f"Total number of combinations: {len(combinations)}")
    index_dict_A = {element: index for index, element in enumerate(nodes_type_A)}
    index_dict_C = {element: index for index, element in enumerate(nodes_type_C)}
    idx_a = [index_dict_A[a] for a, _ in combinations]
    idx_c = [index_dict_C[c] for _, c in combinations]
    all_products_list = []
    print("Start counting in batches! ! !")
    for i in tqdm(range(0, len(combinations), batch_size), desc="Processing batches", unit="batches"):
        batch_idx_a = idx_a[i:i + batch_size]
        batch_idx_c = idx_c[i:i + batch_size]
        batch_idx_a_tensor = torch.tensor(batch_idx_a, device=device)
        batch_idx_c_tensor = torch.tensor(batch_idx_c, device=device)
        batch_products = torch.mul(embeddings_A[batch_idx_a_tensor], embeddings_C[batch_idx_c_tensor])
        batch_products_cpu = batch_products.cpu().numpy()
        all_products_list.append(batch_products_cpu)
    all_products_cpu = np.concatenate(all_products_list, axis=0)
    node_ids = [f'{a}_{c}' for a, c in combinations]
    result_df = pd.DataFrame(all_products_cpu, index=node_ids, columns=[f'feature_{i}' for i in range(embeddings_A.shape[1])])
    result_df = result_df.reset_index()
    result_df = result_df.rename(columns={'index': 'new_index_column'})
    return result_df

def create_feature_dataframe(matrices):
    matrix_values = {name: matrix.values for name, matrix in matrices.items()}
    row_names = list(matrices.values())[0].index
    col_names = list(matrices.values())[0].columns
    three_dimensional_array = np.stack(list(matrix_values.values()), axis=2)
    num_rows, num_cols, num_features = three_dimensional_array.shape
    feature_names = list(matrices.keys())
    combined_features = []
    for i in range(num_rows):
        for j in range(num_cols):
            feature_id = f"{row_names[i]}_{col_names[j]}"
            feature_vector = three_dimensional_array[i, j, :]
            feature_dict = dict(zip(feature_names, feature_vector))
            combined_features.append([feature_id] + list(feature_dict.values()))
    columns = ['feature_id'] + feature_names
    df_features = pd.DataFrame(combined_features, columns=columns)

    return df_features

