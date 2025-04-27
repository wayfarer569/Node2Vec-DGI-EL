import pandas as pd
import torch
import dgl

def create_graph(excel_file):

    node_types_df = pd.read_excel(excel_file, sheet_name='Nodetypes')
    node_types_df['Nodename'] = node_types_df['Nodename'].astype(str)
    node_name_to_type = dict(zip(node_types_df['Nodename'], node_types_df['Nodetype']))
    """
    edges_dfs = {
        edge_key: pd.read_excel(excel_file, sheet_name=edge_key)
        for edge_key in ['Herb-Ingredient', 'Herb-Target', 'Herb-Disease',
                          'Ingredient-Target','Ingredient-Disease',
                          'Disease-Target',
                         'Target1-Target2']}
    """
    edges_dfs = {
        edge_key: pd.read_excel(excel_file, sheet_name=edge_key)
        for edge_key in ['Drug-Disease', 'Drug-Protein', 'Protein-Disease',
                          'Protein1-Protein2']}






    all_nodes = set()
    for edge_key, df in edges_dfs.items():
        if 'src_name' not in df.columns or 'dst_name' not in df.columns:
            raise ValueError(f"sheet:{edge_key} is 'Source' or 'Target' column missing")
        nodes_from_edge = set(pd.concat([df['src_name'].astype(str), df['dst_name'].astype(str)]).tolist())
        all_nodes.update(nodes_from_edge)
    print(f"Number of unique nodes in edge data: {len(all_nodes)}")

    missing_nodes_in_edge_data = [node_name for node_name in all_nodes if node_name not in node_name_to_type]
    if missing_nodes_in_edge_data:
        raise ValueError(f"Node name in edge data '{', '.join(missing_nodes_in_edge_data)}' not found in node data")
    else:
        print("All nodes in edge data are found in node data")
    all_node_names_in_data = set(node_name_to_type.keys())
    missing_nodes_in_node_data = [node_name for node_name in all_node_names_in_data if node_name not in all_nodes]
    if missing_nodes_in_node_data:
        raise ValueError(f"Node name in node data '{', '.join(missing_nodes_in_node_data)}' not found in edge data")
    else:
        print("All nodes in node data are found in edge data")

    valid_edges = set()
    for edge_key, df in edges_dfs.items():
        df['Source'] = df['src_name'].astype(str)
        df['Target'] = df['dst_name'].astype(str)
        valid_edges_df = df[df['Source'].isin(node_name_to_type.keys()) & df['Target'].isin(node_name_to_type.keys())]
        for u, v in zip(valid_edges_df['Source'], valid_edges_df['Target']):
            valid_edges.add((min(u, v), max(u, v)))
    print(f"Number of edges after deduplication: {len(valid_edges)}")

    node_ids = sorted(node_name_to_type.keys())
    node_id_map = {node_name: idx for idx, node_name in enumerate(node_ids)}
    mapped_src_nodes = []
    mapped_dst_nodes = []
    for u, v in valid_edges:
        mapped_src_nodes.append(node_id_map[u])
        mapped_dst_nodes.append(node_id_map[v])
        mapped_src_nodes.append(node_id_map[v])
        mapped_dst_nodes.append(node_id_map[u])
    mapped_src_nodes = torch.tensor(mapped_src_nodes, dtype=torch.int64)
    mapped_dst_nodes = torch.tensor(mapped_dst_nodes, dtype=torch.int64)
    g = dgl.graph((mapped_src_nodes, mapped_dst_nodes))

    edges = list(zip(g.edges()[0].tolist(), g.edges()[1].tolist()))
    edge_set = set(edges)
    missing_reverse_edges = []
    for u, v in edges:
        if (v, u) not in edge_set:
            missing_reverse_edges.append((u, v))
    if missing_reverse_edges:
        print(f"Found {len(missing_reverse_edges)} edges without corresponding reverse edges.")
    else:
        print("All edges have corresponding reverse edges.")
    node_info = {node_id: (node_name, node_name_to_type.get(node_name, 'Unknown')) for node_name, node_id in node_id_map.items()}

    return g,node_info


