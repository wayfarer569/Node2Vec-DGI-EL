import gc
import pickle
import time

import numpy as np
import pandas as pd
import torch

from data_split.graph import create_graph
from data_split.graph_data_preprocessor import load_and_group_nodes_by_type, GraphDataPreprocessor

from Model.dgi      import DGI_GAT_main
from Model.node2vec import Node2Vec_main

from Node_pair_FeatureVectors.all_feature_vectors  import build_edge_info,Hadamard_product,create_feature_dataframe
from Node_pair_FeatureVectors.test_feature_vectors import EdgeFeatureExtractor

from sklearn.model_selection import train_test_split
from Ensemble_Learning_Model.ensemble_learning import EnsembleModel_RF

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_test_data(DGI_emb,Ingredient_nodes,Disease_nodes,g,test_positive_edges,test_negative_edges):

    nodes_type_A = Ingredient_nodes
    nodes_type_C = Disease_nodes

    embeddings_np = torch.from_numpy(DGI_emb.cpu().detach().numpy()).to(dtype=torch.float32)
    embeddings_A = embeddings_np[nodes_type_A].to(device)
    embeddings_C = embeddings_np[nodes_type_C].to(device)
    nodepair_featurevectors = Hadamard_product(embeddings_A,embeddings_C,nodes_type_A,nodes_type_C)
    edge_info =build_edge_info(nodes_type_A,nodes_type_C, g)
    merged_dict_all = {}
    merged_dict_all['edge_info'] = edge_info
    df = create_feature_dataframe(merged_dict_all)
    data = pd.concat([nodepair_featurevectors, df], axis=1)
    extractor = EdgeFeatureExtractor(data)
    data = extractor.extract_features(test_positive_edges, test_negative_edges)
    torch.cuda.empty_cache()
    return data
def test_data_to_train(data,model_save_file,model_save):
        print(data.columns)
        columns_to_drop = ['new_index_column', 'feature_id']
        data = data.drop(columns=columns_to_drop)
        ERF = EnsembleModel_RF()
        X_train, X_test, y_train, y_test = train_test_split(data.drop('edge_info', axis=1), data['edge_info'],test_size=0.3, random_state=42)
        y_train = y_train.astype(int)
        y_test  = y_test.astype(int)
        X_train_with_y_train= X_train.assign(y=y_train)
        positive_class = X_train_with_y_train[X_train_with_y_train['y'] == 1]
        negative_class = X_train_with_y_train[X_train_with_y_train['y'] == 0]
        n_subsets = round(len(negative_class)/len(positive_class))
        negative_subsets =  np.array_split(negative_class, n_subsets)
        final_prediction= ERF.train_and_predict_cpu(positive_class, negative_subsets, n_subsets, X_train, y_train, X_test,model_save_file,model_save)
        ROC_best_threshold = ERF.find_best_threshold_ROC(y_test, final_prediction)
        print("ROC_best_threshold",ROC_best_threshold)
        ERF.evaluate_model(y_test, final_prediction,ROC_best_threshold)
        PR_best_threshold = ERF.find_best_threshold_PR(y_test, final_prediction)
        print("AUPR_best_threshold",PR_best_threshold)
        ERF.evaluate_model(y_test, final_prediction,PR_best_threshold)

def True_edge_matrix(node_info,nodes_type_A,nodes_type_C,g):
    node_id_to_name = {node_id: node_name for node_id, (node_name, node_type) in node_info.items()}
    edge_info_df = build_edge_info(nodes_type_A, nodes_type_C, g)
    edge_info_df.index = edge_info_df.index.map(node_id_to_name.get)
    edge_info_df.columns = edge_info_df.columns.map(node_id_to_name.get)
    return edge_info_df

def main():
    start_time = time.time()
    ### data split
    data_file = 'data/data.xlsx'

    g,node_info = create_graph(data_file)
    print("graph",g)
    hide_percentage    = 0.3
    Need_to_use_test = True

    ### Node2Vec
    Node2Vec_emb_save  = False
    Node2Vec_emb_txt_file = "result/node2vec_emb.txt"
    Node2Vec_emb_npy_file = "result/node2vec_emb.npy"
    Node2Vec_emb_dim   = 128
    p = 0.5
    q = 2
    walk_length    = 80
    num_walks      = 10
    workers        = 1
    window         = 10

    ### DGI
    DGI_emb_save   = True
    DGI_emb_file   = "result/DGI_emb.npy"
    DGI_model_file = 'result/DGI_best_model.pkl'
    DGI_hid_dim   = 192
    DGI_out_dim    = 128
    DGI_lr         = 0.001
    DGI_att_head   = 4
    DGI_epoch      = 200

    ### Ensemble
    EL_model_save      = True
    EL_model_save_file = 'result/ensemble_models.joblib'


    # --------------------------------data split---------------------------------------------------
    node_type_map = load_and_group_nodes_by_type(node_info)
    Ingredient_nodes = node_type_map.get('Ingredient', [])
    Disease_nodes    = node_type_map.get('Disease', [])
    graph_data_preprocessor = GraphDataPreprocessor(g, Ingredient_nodes,Disease_nodes,hide_percentage)
    train_graph, test_positive_edges, test_negative_edges = graph_data_preprocessor.prepare_train_and_test_data()
    print("train_graph",train_graph)

    if Need_to_use_test:
        with open('result/nodes_type_Ingredient.pkl', 'wb') as f:
            pickle.dump(Ingredient_nodes, f)
        with open('result/nodes_type_Disease.pkl', 'wb') as f:
            pickle.dump(Disease_nodes, f)
        with open('result/node_info.pkl', 'wb') as f:
            pickle.dump(node_info, f)
        edge_info_df = True_edge_matrix(node_info,Ingredient_nodes,Disease_nodes,g)
        edge_info_df.to_csv("result/edge_info_with_names.csv", encoding='utf-8-sig')

    #-------------------------------Node2vec-------------------------------------------------------

    Node2Vec_embeddings =Node2Vec_main(train_graph,Node2Vec_emb_save,Node2Vec_emb_txt_file,Node2Vec_emb_npy_file,
                                       Node2Vec_emb_dim,walk_length,num_walks,p,q,workers,window)

    #------------------------------DGI--------------------------------- -----------------------

    print(f"Using device: {device}")
    feature_tensor = torch.from_numpy(Node2Vec_embeddings).float()
    train_graph.ndata['feat'] = feature_tensor
    DGI_embeddings_np = DGI_GAT_main(train_graph,DGI_emb_save,DGI_emb_file,DGI_model_file,DGI_hid_dim,
                                     DGI_out_dim,DGI_att_head,DGI_epoch,DGI_lr)

    #------------------------------Ensemble--------------------------------------------- ----------

    data = get_test_data(DGI_embeddings_np,Ingredient_nodes,Disease_nodes,g,test_positive_edges,test_negative_edges)
    test_data_to_train(data,EL_model_save_file,EL_model_save)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"code running time: {elapsed_time:.4f}seconds")

if __name__ == '__main__':
    main()





