import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from lightgbm import LGBMClassifier

# --- データ読み込み ---
DATA_PATH = "data/date_bingo5.csv"
df = pd.read_csv(DATA_PATH)

# --- 出現数字列 ---
number_cols = [f"数字{i+1}" for i in range(8)]

# --- 特徴量作成関数 ---
def create_features(df):
    features = []
    for _, row in df[number_cols].iterrows():
        nums = row.values.astype(int)
        vec = [1 if i in nums else 0 for i in range(1, 41)]
        features.append(vec)
    return pd.DataFrame(features, columns=[f"feature_{i}" for i in range(1, 41)])

# --- 入力と出力を1回ずらす（未来予測） ---
X = create_features(df)
X_shifted = X[:-1].reset_index(drop=True)
y_shifted = X[1:].reset_index(drop=True)

# --- 学習データ分割 ---
X_train, X_test, y_train, y_test = train_test_split(X_shifted, y_shifted, test_size=0.2, random_state=42)

# --- モデル学習 ---
model = MultiOutputClassifier(LGBMClassifier(random_state=42))
model.fit(X_train, y_train)

# --- 保存フォルダ作成 ---
os.makedirs("model", exist_ok=True)

# --- モデル保存 ---
with open("model/bingo5_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ モデルの学習と保存が完了しました: model/bingo5_model.pkl")
