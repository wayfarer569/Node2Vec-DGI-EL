requirements :
    python == 3.9.20
    torch ==  1.8.1+cu111
    dgl-cu111 == 0.6.1
    networkx == 3.2.1
    node2vec == 0.5.0
    numpy  == 1.26.4
    pandas == 1.3.5

Introduction :
    We create a dataset based on the ETCM and String Database, which includes herb-ingredient-target-disease relationships, and build a Node2Vec-DGI-EL model for ingredient-disease prediction.

dataset :
    Please build the dataset based on the data - data.xlsx file and adjust the edges_df of data_split--graph.py according to the actual sheet_name

run code :
     Train the model :
        main.py

     Use the model for precision :
        please set relevant parameters in main.py :
            Need_to_use_test = True
            DGI_emb_save     = True
            EL_model_save    = True
        after:
            test.py
            The adjacency matrix of Ingredient-Disease is stored in the edge_info_with_names.csv file in the result folder.