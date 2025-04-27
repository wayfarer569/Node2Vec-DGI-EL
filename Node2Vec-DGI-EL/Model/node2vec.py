from node2vec import Node2Vec
import numpy as np
def save_embeddings_to_file(embeddings, filename_txt, filename_npy):
    """
    将节点 ID 和嵌入向量按照节点 ID 的顺序排列后保存到文件
    :param embeddings: 嵌入向量对象（来自 Node2Vec 的 model.wv）
    :param filename_txt: 保存节点 ID 和嵌入向量到 .txt 文件
    :param filename_npy: 保存嵌入向量到 .npy 文件
    """
    node_embeddings = {node: embeddings[node] for node in embeddings.index_to_key}
    sorted_nodes = sorted(node_embeddings.keys(), key=lambda x: int(x))
    sorted_node_embeddings = {node: node_embeddings[node] for node in sorted_nodes}
    with open(filename_txt, "w") as f:
        for node, embedding in sorted_node_embeddings.items():
            f.write(f"{node} {' '.join(map(str, embedding))}\n")
    print(f"Node IDs and embeddings saved to {filename_txt} (sorted by node IDs)")
    sorted_embeddings_array = np.array([node_embeddings[node] for node in sorted_nodes])
    np.save(filename_npy, sorted_embeddings_array)
    print(f"Sorted embeddings saved to {filename_npy}")

def Node2Vec_main(G,save,file1,file2,embedding_dimension,walk_length,num_walks,p,q,workers,window):
    G = G.to_networkx()
    G = G.to_undirected()
    node2vec  = Node2Vec(G, dimensions=embedding_dimension, walk_length=walk_length, num_walks=10, p=p, q=2, workers=workers)
    model = node2vec.fit(window=window)
    embeddings = model.wv
    if save is True:
        save_embeddings_to_file(embeddings,file1,file2)
    return embeddings.vectors