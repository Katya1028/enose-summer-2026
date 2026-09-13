import pandas as pd
from pathlib import Path


def load_csv(file_path):
    data = pd.read_csv(file_path)
    return data

def find_labeled_csv_files(root_folder):
    root = Path(root_folder)

    files_with_labels = []

    # 找root_folder 底下的每個類別資料夾
    for class_folder in root.iterdir():

        if class_folder.is_dir():
            label = class_folder.name

            #找這個類別資料夾底下所有的csv
            for file_path in class_folder.glob("*.csv"):
                files_with_labels.append((file_path, label))
    return files_with_labels