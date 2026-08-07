from pathlib import Path
import glob
import csv

folder = Path(input("請輸入資料夾: "))

files = glob.glob(str(folder / "*.csv"))

summary = []

for file in files:
    try:
        with open(file, encoding = "utf-8") as document:
            reader = csv.reader(document)
            header = next(reader)
            count = 0
            for row in reader:
                count+=1
        file_name = Path(file).name

        print("=" *30)
        print("檔案名稱: ", file_name)
        print("列數: ", count)
        print("欄位名稱: ", header)

        summary.append([
            file_name,
            count,
            ",".join(header)
])
    except Exception as error:
        print("=" *30)
        print(file, "檔案讀取失敗")
        print("錯誤原因: ", error)

with open("summary.csv", "w", newline="", encoding= "utf-8") as document:
    writer = csv.writer(document)

    writer.writerow([
        "檔案名稱",
        "列數",
        "欄位名稱"
    ])

    writer.writerow(summary)

print("=" *30)
print("輸出完成")