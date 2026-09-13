import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv

from pathlib import Path
def extract_features(data, file_path, label): #定義特徵函數(檔案名稱，分類)

    features = {}

    #將時間轉成從0秒開始
    data["datetime"] = pd.to_datetime(data["datetime"])
    data["time_sec"] = (data["datetime"]-data["datetime"].iloc[0]).dt.total_seconds() #先將資料的時間修到剩秒數

    #先找出response區間和response的開始時間
    response = data[data["step"] == "Measure"]
    response_start = response["time_sec"].min()
    response_end = response["time_sec"].max()

#基線校正

    #先切出baseline區間
    baseline = data[(data["time_sec"]>=response_start-10) & (data["time_sec"]<response_start)] #取baseline為進氣前10秒

    for i in range(1,17):
        channel = f"ch{i}" #用f即可自動將變數放到字串裡
        corrected_channel = f"{channel}_corrected" #令修正完時間後的channel為corrected_channel
        baseline_mean = baseline[channel].mean() #將baseline取平均
        data[corrected_channel] = data[channel]-baseline_mean #全曲線-baseline

    #baseline校正完後重新切區間
    response = data[data["step"] == "Measure"]
    recovery = data[data["step"] == "Clean"]
    recovery_start = recovery["time_sec"].min()


#特徵設計
    #delta_f: response 區間的穩態頻率變化量
    #設穩態值為response前後各切10秒的平均

    steady = response[(response["time_sec"]>=response_start+10)&(response["time_sec"]<response_end-10)] #在response內量測到的頻率設為穩態區間

    for i in range(1,17):
        channel = f"ch{i}"
        corrected_channel = f"{channel}_corrected"
        delta_f = steady[corrected_channel].mean() #將穩態區間的平均變化量取平均()
        features[f"{channel}_delta_f"] = float(delta_f) #轉成浮點數

    #response_slope: 反應初期的斜率
    #在response初期，QCM校正後頻率變化率；在晶體吸附VOC後，共振頻率下降，斜率為負數
    response_initial = response[(response["time_sec"]>=response_start +10)&(response["time_sec"]<response_start+20)] #取response開始穩定後的前10秒數據
    for i in range(1,17):
        channel = f"ch{i}"
        corrected = f"{channel}_corrected"

        x_response = response_initial["time_sec"] 
        y_response = response_initial[corrected]
        response_slope, _ = np.polyfit(x_response, y_response, 1)  #算出response的斜率和截距

        features[f"{channel}_response_slope"] = float(response_slope) 
    

    #recovery_slope: 恢復期的斜率
    #在recovery初期，頻率變化率；原本吸附的VOC離開感測層，即為感測層的恢復速度，斜率為正數
    recovery_initial = recovery[(recovery["time_sec"]>=recovery_start+10)&(recovery["time_sec"]<recovery_start+20)] #取recovery開始穩定後的前10秒數據
    for i in range(1,17):
        channel = f"ch{i}"
        corrected = f"{channel}_corrected"

        x_recovery = recovery_initial["time_sec"]
        y_recovery = recovery_initial[corrected]
        recovery_slope, _ = np.polyfit(x_recovery, y_recovery, 1) #算出recovery的斜率和截距
        features[f"{channel}_recovery_slope"] = float(recovery_slope)


    #AUC: response區間的曲線下面積
    #有效的反應區間累積的反應量
    for i in range(1,17):
        channel = f"ch{i}"
        corrected = f"{channel}_corrected"
        auc = np.trapezoid(steady[corrected], steady["time_sec"]) #利用trapezoid函數算出曲線下面積
        features[f"{channel}_auc"] = float(auc)

    #算t90: 達到90%穩態值所需時間
    #這顆sensor要花多久，才能"幾乎完成"他的response?
    for i in range(1,17):
        channel = f"ch{i}"
        corrected = f"{channel}_corrected"
        steady_value = steady[corrected].mean() #將在穩態區間的頻率取平均
        threshold_90 = steady_value*0.9 #標準為90%

        #判斷steady_value正負(因為response有可能是正負)
        #reach_90: 所有有達到90%的資料
        if steady_value <0: 
            reach_90 = response[response[corrected] <= threshold_90] 
        else: 
            reach_90 = response[response[corrected] >= threshold_90]

        response_start = response["time_sec"].iloc[0] #取response開始的第一筆資料
        
        if len(reach_90)>0: #如果剛剛取到的資料數量有大於0 (看有沒有成功找到資料)
            t90_time = reach_90["time_sec"].iloc[0] #有的話就取第一筆資料!
            t90 = t90_time-response_start #拿到的秒數減掉response開始的時間，就可以知道是第幾秒達到90%
        else:
            t90 = np.nan#沒有取道的話就回傳NaN
            
        features[f"{channel}_t90"] = float(t90)

    path = Path(file_path) #從檔案路徑自動取得sample資訊
    sample_id = path.stem #.stem取得檔案名稱，不要副檔名
    subject_id = sample_id.split("_")[0] #遇到"_"就切開來，形成一個list，取裡面的第一個

    features["subject_id"] = subject_id #把剛剛得到的都放到字典裡
    features["sample_id"] = sample_id
    features["label"] = label

    return features #回傳給features這個字典