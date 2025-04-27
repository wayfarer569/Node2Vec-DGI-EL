import torch
import numpy as np
import pandas as pd
import joblib
import logging
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def find_node_by_data(node_info,data_value):
    for node_id, (data, node_type) in node_info.items():
        if data == data_value:
            return node_id, node_type
    return None, None

def get_formatted_data(Node_embedding_vector,node_info,Ingredient_id,nodes_type_C):
    node_id, node_type = find_node_by_data(node_info,Ingredient_id)
    print(node_id,node_type)
    Ingredient_vector = Node_embedding_vector[node_id]

    result_list = []
    index_list = []
    for i in nodes_type_C:
        product = Ingredient_vector * Node_embedding_vector[i]
        result_list.append(product)
        index_name = f"{node_id}_{i}"
        index_list.append(index_name)
    formatted_df = pd.DataFrame(result_list, index=index_list)
    new_columns = [f'feature_{col}' for col in formatted_df.columns]
    formatted_df.columns = new_columns
    return formatted_df

def prediction(formatted_df,node_info,models,logistic_regression):
    index = formatted_df.index
    split_names = [i.split('_') for i in index]
    drug_disease_info = []
    for name in split_names:
        drug = node_info[int(name[0])][0]
        disease = node_info[int(name[1])][0]
        node_type = node_info[int(name[1])][1]
        drug_disease_info.append((name,drug, disease, node_type))
    df_drug_disease = pd.DataFrame(drug_disease_info,columns=['node_pair', 'drug', 'disease', 'node_type'])
    feature_data = formatted_df.iloc[:, :]
    model_predictions = np.array([model.predict_proba(feature_data)[:, 1] for model in models]).T
    final_predictions = logistic_regression.predict_proba(model_predictions)[:, 1]

    combined_full_data = pd.concat([df_drug_disease.reset_index(drop=True),
                                    pd.DataFrame(final_predictions.reshape(-1, 1), columns=["final_predictions"])], axis=1)
    return combined_full_data


def main():

    Ingredient_prediction_disease = False
    Disease_prediction_ingredient = True

    Node_embedding_vector_file = 'result/DGI_emb.npy'
    node_info_file = 'result/node_info.pkl'
    EL_model_file  = 'result/ensemble_models.joblib'

    with open(node_info_file, 'rb') as f:
        node_info = pickle.load(f)

    Node_embedding_vector= np.load(Node_embedding_vector_file)

    with open(EL_model_file, 'rb') as f:
        model_dict = joblib.load(f)
        models = model_dict['models']
        logistic_regression = model_dict['logistic_regression']



    if Ingredient_prediction_disease is True :
        disease_node_file  = 'result/nodes_type_Disease.pkl'
        with open(disease_node_file, 'rb') as f:
            Disease_nodes = pickle.load(f)
        Ingredient_id = "Salvianolic Acid B"
        data = get_formatted_data(Node_embedding_vector,node_info,Ingredient_id,Disease_nodes)
        combined_full_data = prediction(data,node_info,models,logistic_regression)
        combined_full_data.to_csv(f"{Ingredient_id}_result.csv", index=False)


    if Disease_prediction_ingredient is True :
        Ingredient_node_file  = 'result/nodes_type_Ingredient.pkl'
        with open(Ingredient_node_file, 'rb') as f:
            Ingredient_nodes = pickle.load(f)
        Disease_id = "Lung cancer"
        data = get_formatted_data(Node_embedding_vector,node_info,Disease_id,Ingredient_nodes)
        combined_full_data = prediction(data,node_info,models,logistic_regression)
        combined_full_data.to_csv(f"{Disease_id}_result.csv", index=False)


if __name__ == "__main__":
    main()
