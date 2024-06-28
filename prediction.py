from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pickle
import pandas as pd

app = FastAPI()

# モデルのロード
Amodel = pickle.load(open("./models/Amodel.pkl", "rb"))
Lmodel = pickle.load(open("./models/Lmodel.pkl", "rb"))
Pmodel = pickle.load(open("./models/Pmodel.pkl", "rb"))

class UserInfo(BaseModel):
    APreviousDayCompletion: float
    APreviousDayTarget: float
    AWeeklyCompletion: float
    Age: int
    Frequency: str
    Gender: str
    Goal: str
    Height: int
    LPreviousDayCompletion: float
    LPreviousDayTarget: float
    LWeeklyCompletion: float
    PPreviousDayCompletion: float
    PPreviousDayTarget: float
    PWeeklyCompletion: float
    Weight: float

def parsonSelection(data):
    # 目標BMIに基づく理想体重の算出
    goal_bmi = {
        "Male": {"MuscleStrength": 25.0, "WeightLoss": 22.0, "HealthMaintenance": 24.0},
        "Female": {"MuscleStrength": 23.0, "WeightLoss": 20.0, "HealthMaintenance": 22.0},
        "Other": {"MuscleStrength": 24.0, "WeightLoss": 21.0, "HealthMaintenance": 23.0}
    }

    # 辞書から必要な情報を取得
    Gender = data["Gender"]
    Frequency = data["Frequency"]
    Age = data["Age"]
    Goal = data["Goal"]
    Height = data["Height"]
    Weight = data["Weight"]

    # 目標BMIに基づく理想体重の算出
    ideal_bmi = goal_bmi[Gender][Goal]
    idealWeight = round(ideal_bmi * (Height / 100) ** 2, 1)

    # 文字列を数値に置換
    gender_mapping = {"Male": 1, "Female": 2, "Other": 3}
    goal_mapping = {"MuscleStrength": 1, "WeightLoss": 2, "HealthMaintenance": 3}
    frequency_mapping = {"Low": 1, "Moderate": 2, "High": 3}
    Gender, Goal, Frequency = map(lambda x, mapping: mapping.get(x, x), [Gender, Goal, Frequency], [gender_mapping, goal_mapping, frequency_mapping])

    # A
    APreviousDayCompletion = data["APreviousDayCompletion"]
    AWeeklyCompletion = data["AWeeklyCompletion"]
    APreviousDayTarget = data["APreviousDayTarget"]
    # L
    LPreviousDayCompletion = data["LPreviousDayCompletion"]
    LWeeklyCompletion = data["LWeeklyCompletion"]
    LPreviousDayTarget = data["LPreviousDayTarget"]
    # P
    PPreviousDayCompletion = data["PPreviousDayCompletion"]
    PWeeklyCompletion = data["PWeeklyCompletion"]
    PPreviousDayTarget = data["PPreviousDayTarget"]

    # MinMaxScalerのインスタンスを作成
    scaler = MinMaxScaler()
    # 選択した列のデータを正規化
    new_data = np.array([[Gender, Frequency, Age, Goal, Height, Weight, idealWeight, APreviousDayCompletion, AWeeklyCompletion, APreviousDayTarget, LPreviousDayCompletion, LWeeklyCompletion, LPreviousDayTarget, PPreviousDayCompletion, PWeeklyCompletion, PPreviousDayTarget]], dtype=np.float32)

    # 特徴量の名前を指定してDataFrameを作成
    columns = ["Gender", "Frequency", "Age", "Goal", "Height", "Weight", "IdealWeight", "APreviousDayCompletion", "AWeeklyCompletion", "APreviousDayTarget", "LPreviousDayCompletion", "LWeeklyCompletion", "LPreviousDayTarget", "PPreviousDayCompletion", "PWeeklyCompletion", "PPreviousDayTarget"]
    df_data = pd.DataFrame(scaler.fit_transform(new_data), columns=columns)

    # 予測＆格納
    TargetRepsDict = {}
    models = {"A": Amodel, "L": Lmodel, "P": Pmodel}
    category = {"A": "AbsTraining", "L": "LegTraining", "P": "PectoralTraining"}
    for model in models:
        # 推論
        prediction = models[model].predict(df_data)[0]
        # 推論結果を格納
        TargetRepsDict[category[model]] = round(prediction, 0)

    return TargetRepsDict

@app.post("/target/")
def bmi_prediction(req: UserInfo):
    data = req.dict()
    
    # 関数を呼び出して結果を取得
    result = parsonSelection(data)
    
    return result