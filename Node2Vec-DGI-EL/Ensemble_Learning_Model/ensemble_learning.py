import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score,\
    accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve, roc_curve,average_precision_score
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from joblib import Parallel, delayed
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from joblib import Memory
from joblib import dump


memory = Memory(r"E:\joblib_cache", verbose=0)

class EnsembleModel_RF:
    def __init__(self):

        self.models = None
        self.logistic_regression = None


    def train_model(self, subset_idx, positive_class, negative_subsets):
        positive_class = positive_class
        subset = negative_subsets[subset_idx]
        balanced_data = pd.concat([positive_class, subset])
        X_balanced = balanced_data.iloc[:, :-1]
        y_balanced = balanced_data.iloc[:, -1]
  
        model = RandomForestClassifier(
            n_estimators=100,
            criterion='gini',
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='sqrt',
            bootstrap=True,
            oob_score=False,
            n_jobs=10,
            random_state=42
        )
        model.fit(X_balanced, y_balanced)
        print(f"Model for subset {subset_idx} trained")
        return model

    def train_ensemble(self, positive_class, negative_subsets, n_subsets):
        models = Parallel(n_jobs=15,prefer="threads")(
                delayed(self.train_model)(i, positive_class, negative_subsets)
                for i in range(n_subsets))
        return models

    def train_and_predict_cpu(self, positive_class, negative_subsets, n_subsets, X_train, y_train, X_test,model_save_file,model_save=False):

        models = self.train_ensemble(positive_class, negative_subsets, n_subsets)
        print("RF models is done")

        base_predictions_train = np.array([model.predict_proba(X_train)[:, 1] for model in models]).T
        print("open logistic_regression train")
        logistic_regression = LogisticRegression(class_weight='balanced')
        logistic_regression.fit(base_predictions_train, y_train)
        print("logistic_regression train is done")
        if model_save:
            model_dict = {'models': models, 'logistic_regression': logistic_regression}
            base_predictions_test = np.array([model.predict_proba(X_test)[:, 1] for model in models]).T

            final_prediction = logistic_regression.predict_proba(base_predictions_test)[:, 1]
            dump(model_dict, model_save_file)
            return final_prediction
        else:
            base_predictions_test = np.array([model.predict_proba(X_test)[:, 1] for model in models]).T
            final_prediction = logistic_regression.predict_proba(base_predictions_test)[:, 1]
            return final_prediction


    def find_best_threshold_ROC(self, y_test, final_prediction):

        fpr, tpr, thresholds = roc_curve(y_test, final_prediction)

        J = tpr - fpr
        best_threshold_index = np.argmax(J)
        best_threshold = thresholds[best_threshold_index]
        return best_threshold

    def find_best_threshold_PR(self, y_test, final_prediction):
        precision, recall, thresholds = precision_recall_curve(y_test, final_prediction)
        f1_scores = 2 * (precision * recall) / (precision + recall)
        best_threshold_index = np.argmax(f1_scores)
        best_threshold = thresholds[best_threshold_index]
        return best_threshold


    def evaluate_model(self, y_test, final_prediction, threshold):

        y_pred = (final_prediction > threshold).astype(int)

        print("Classification Report:")
        print(classification_report(y_test, y_pred))

        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

        pos_precision = precision_score(y_test, y_pred)
        pos_recall = recall_score(y_test, y_pred)
        pos_f1 = f1_score(y_test, y_pred)

        neg_precision = precision_score(y_test, y_pred, pos_label=0)
        neg_recall = recall_score(y_test, y_pred, pos_label=0)
        neg_f1 = f1_score(y_test, y_pred, pos_label=0)

        macro_precision = precision_score(y_test, y_pred, average='macro')
        macro_recall = recall_score(y_test, y_pred, average='macro')
        macro_f1 = f1_score(y_test, y_pred, average='macro')

        weighted_precision = precision_score(y_test, y_pred, average='weighted')
        weighted_recall = recall_score(y_test, y_pred, average='weighted')
        weighted_f1 = f1_score(y_test, y_pred, average='weighted')

        print(f"Positive Class Precision: {pos_precision}")
        print(f"Positive Class Recall: {pos_recall}")
        print(f"Positive Class F1 Score: {pos_f1}")
        print(f"Negative Class Precision: {neg_precision}")
        print(f"Negative Class Recall: {neg_recall}")
        print(f"Negative Class F1 Score: {neg_f1}")

        print(f"Macro Average Precision: {macro_precision}")
        print(f"Macro Average Recall: {macro_recall}")
        print(f"Macro Average F1 Score: {macro_f1}")

        print(f"Weighted Average Precision: {weighted_precision}")
        print(f"Weighted Average Recall: {weighted_recall}")
        print(f"Weighted Average F1 Score: {weighted_f1}")

        print(f"ROC-AUC Score: {roc_auc_score(y_test, final_prediction)}")
        print(f"Accuracy Score: {accuracy_score(y_test, y_pred)}")
        print(f"Average Precision Score: {average_precision_score(y_test, final_prediction)}")


