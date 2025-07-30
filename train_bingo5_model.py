import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from lightgbm import LGBMClassifier

# CSV読み込み
DATA_PATH = "data/date_bingo5.csv"
df = pd.read_csv(DATA_PATH)

# 出現数字列（8列）
number_cols = [f"数字{i+1}" for i in range(8)]

# 特徴量Xの作成：出現履歴のワンホット（40個の数字について1 or 0）
def create_features(df):
    features = []
    for _, row in df[number_cols].iterrows():
        nums = row.values.astype(int)
        vec = [1 if i in nums else 0 for i in range(1, 41)]
        features.append(vec)
    return pd.DataFrame(features, columns=[f"num_{i}" for i in range(1, 41)])

X = create_features(df)

# 1回先のデータを予測対象にするために、1つシフト
X_shifted = X[:-1].reset_index(drop=True)
y_shifted = X[1:].reset_index(drop=True)

# 学習データとテストデータに分割
X_train, X_test, y_train, y_test = train_test_split(X_shifted, y_shifted, test_size=0.2, random_state=42)

# モデル定義（LightGBM × MultiOutputClassifier）
model = MultiOutputClassifier(LGBMClassifier())
model.fit(X_train, y_train)

# 保存ディレクトリ作成
os.makedirs("model", exist_ok=True)

# モデル保存
with open("model/bingo5_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ モデル学習と保存が完了しました: model/bingo5_model.pkl")
