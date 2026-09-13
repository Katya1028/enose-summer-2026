import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report

def prepare_data(data):

    X = data.drop(columns=["sample_id","subject_id","label"]) #給模型看的特徵
    y = data["label"] #真正的分類答案

    #同一尿液樣本的分組資訊
    groups=( 
        data["label"].astype(str)
        + "_"
        +data["subject_id"].astype(str)
    )

    return X, y, groups

def run_pca(X):
    #標準化features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    #將80為資料降成2維
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    return X_pca, pca

def plot_pca(X_pca, y, output_path="pca.png"):
    #把PCA結果整理成DataFrame，方便畫圖
    pca_df = pd.DataFrame(
        X_pca,
        columns=["PC1", "PC2"]
    )

    #加上每筆資料真正的label
    pca_df["label"] = y.values

    for label in pca_df["label"].unique():
        group = pca_df[pca_df["label"] == label]

        plt.scatter(
            group["PC1"],
            group["PC2"],
            label=label
        )
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA")
    plt.legend()

    plt.savefig(output_path)
    plt.close()

def create_models():

    #LDA
    lda_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearDiscriminantAnalysis())
    ])

    #SVM
    svm_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC(kernel="linear"))
    ])

    #RandomForest
    rf_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestClassifier(
            n_estimators=100,
            random_state=42
        ))
    ])

    models = {
        "LDA": lda_pipeline,
        "SVM": svm_pipeline,
        "Random Forest": rf_pipeline
    }

    return models

def evaluate_models(X, y, groups, models, n_splits=4):

    #建立GroupKFold 
    gkf = GroupKFold(n_splits = n_splits)

    results=[]

    #一次評估LDA，SVM，RandomForest
    for model_name, model in models.items():

        fold_accuracies = []
        fold_f1_scores = []

        #每一次迴圈就是其中一個fold
        for train_index, test_index in gkf.split(X, y, groups):

            #根據index切出training/ testing
            X_train= X.iloc[train_index]
            X_test = X.iloc[test_index]

            y_train = y.iloc[train_index]
            y_test = y.iloc[test_index]

            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)

            macro_f1 = f1_score(
                y_test, 
                y_pred,
                average="macro",
                zero_division = 0
            )

            fold_accuracies.append(accuracy)
            fold_f1_scores.append(macro_f1)

        results.append({
            "Model": model_name,
            "Accuracy": np.mean(fold_accuracies),
            "Macro_F1": np.mean(fold_f1_scores)
        })

    return pd.DataFrame(results)

def evaluate_best_model(X, y, groups, model, n_splits=4):

    gkf = GroupKFold(n_splits=n_splits)

    all_true = []
    all_pred = []

    for train_index, test_index in gkf.split(X, y, groups):

        X_train = X.iloc[train_index]
        X_test = X.iloc[test_index]

        y_train = y.iloc[train_index]
        y_test = y.iloc[test_index]

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        all_true.extend(y_test)
        all_pred.extend(y_pred)

    return all_true, all_pred

def plot_confusion_matrix(
        all_true,
        all_pred,
        output_path= "confusion_matrix.png"
):
    labels =sorted(set(all_true))

    cm =confusion_matrix(
        all_true,
        all_pred,
        labels=labels
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=labels
    )

    disp.plot()
    plt.title("Confusion Matrix")

    plt.savefig(output_path)
    plt.close()

def get_classification_report(all_true, all_pred):

    report = classification_report(
        all_true, 
        all_pred,
        output_dict = True,
        zero_division = 0
    )

    report_df = pd.DataFrame(report).transpose()

    return report_df