import pandas as pd
from features import extract_features
from data_io import load_csv
from data_io import find_labeled_csv_files
from models import (prepare_data, run_pca, plot_pca, 
                    create_models, evaluate_models, evaluate_best_model,
                    plot_confusion_matrix, get_classification_report)

from pathlib import Path

#將原始資料所在的資料夾命名為"All"
BASE_DIR = Path(__file__).resolve().parent
root_folder = BASE_DIR/ "All"

#分析結果要存的資料夾
output_folder = BASE_DIR/ "outputs"
output_folder.mkdir(exist_ok = True)  #mkdir=make directory

files_with_labels = find_labeled_csv_files(root_folder)

all_features = []

for file_path,label in files_with_labels:
    #讀csv
    data = load_csv(file_path)
    #擷取features
    features = extract_features(data, file_path, label)
    #加入總表
    all_features.append(features)
    #確認程式處理到哪
    #print(file_path, "->", label)

#將得到的features存成表格
feature_table = pd.DataFrame(all_features)

#將subject_id、sample_id、label移到表格最前面
first_columns = ["subject_id", "sample_id", "label"]
#找出剩下的feature欄位
other_columns = [
    col for col in feature_table.columns
    if col not in first_columns
]
#重新排列欄位順序
feature_table = feature_table[
    first_columns + other_columns
]
#存成csv
feature_table.to_csv(output_folder/"all-features.csv", index=False)

#檢查目前有幾個不同的分類
num_classes = feature_table["label"].nunique()

print("Number of classes: ", num_classes)

#如果只有一個類別，就不做分類模型
if num_classes<2:
    print("only one class was found.")
    print("Feature extraction is completed.")
    print("Model classification will be skipped.")
    exit()

#畫成PCA
X, y, groups = prepare_data(feature_table)
X_pca, pca = run_pca(X)
plot_pca(X_pca, y, output_folder/"pca.png")
print("PCA shape: ", X_pca.shape)
print("Explained variance: ", pca.explained_variance_ratio_)
print("X shape: ", X.shape)
print("y shape: ", y.shape)
print("groups shape: ", groups.shape)


print(feature_table["label"].value_counts())
print(feature_table["subject_id"].nunique())

#跑LDA, SVM, RandomForest 模型
models = create_models()

#印出跑出來的結果
results = evaluate_models(
    X,
    y,
    groups,
    models,
    n_splits=4
)
print("\nModel results:")
print(results)

#自動找Accuracy最好的模型
best_model_name = results.loc[
    results["Accuracy"].idxmax(),
    "Model"
]
print("\nBest model:", best_model_name)

#從models裡取得最佳模型
best_model = models[best_model_name]

#收最佳模型的out-of-fold predictions
all_true, all_pred = evaluate_best_model(
    X,
    y,
    groups,
    best_model,
    n_splits=4
)
print("Number of predictions:", len(all_pred))

#畫 confusion matrix
plot_confusion_matrix(
    all_true,
    all_pred,
    output_folder/"confusion_matrix.png"
)

#產生classification report
report = get_classification_report(
    all_true,
    all_pred
)
print("\nClassification Report:")
print(report)

#儲存三個模型的比較結果
results.to_csv(
    output_folder/"model_results.csv",
    index=False
)

#儲存最佳模型的 classification report
report.to_csv(
    output_folder/ "classification_report.csv"
)