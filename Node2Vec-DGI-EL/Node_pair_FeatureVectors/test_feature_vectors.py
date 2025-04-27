import pandas as pd
import numpy as np

class EdgeFeatureExtractor:
    def __init__(self, df_features):
        self.df_features = df_features
        self.df_features.set_index('feature_id', inplace=True)
    def get_feature_vector(self, src, dst):
        feature_id = f"{src}_{dst}"
        try:
            feature_row = self.df_features.loc[feature_id]
            return feature_row.values, feature_id
        except KeyError:
            #print(f"Feature not found: {feature_id}")
            return None

    def extract_features(self, test_positive_edges, test_negative_edges):
        positive_features = []
        for src, dst in test_positive_edges:
            feature_vector = self.get_feature_vector(src, dst)
            if feature_vector is not None:
                positive_features.append(feature_vector)
        negative_features = []
        for src, dst in test_negative_edges:
            feature_vector = self.get_feature_vector(src, dst)
            if feature_vector is not None:
                negative_features.append(feature_vector)
        print("number of positive_features", len(positive_features))
        print("number of negative_features", len(negative_features))

        positive_feature_vectors = [feature[0] for feature in positive_features]
        positive_feature_ids = [feature[1] for feature in positive_features]

        negative_feature_vectors = [feature[0] for feature in negative_features]
        negative_feature_ids = [feature[1] for feature in negative_features]

        positive_feature_vectors = np.array(positive_feature_vectors)
        negative_feature_vectors = np.array(negative_feature_vectors)

        print("positive_features shape:", positive_feature_vectors.shape)
        print("negative_features shape:", negative_feature_vectors.shape)

        combined_features = np.vstack((positive_feature_vectors, negative_feature_vectors))
        combined_feature_ids = positive_feature_ids + negative_feature_ids

        feature_columns = self.df_features.columns.tolist()
        df = pd.DataFrame(combined_features, columns=feature_columns)
        df['feature_id'] = combined_feature_ids

        return df