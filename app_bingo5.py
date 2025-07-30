import streamlit as st
import pandas as pd
import numpy as np
import os
import lightgbm as lgb
import joblib
from datetime import datetime

# ===== データ読み込みと整形 =====
@st.cache_data
def load_data():
    df = pd.read_csv("data/date_bingo5.csv")
    df = df.rename(columns={"抽せん日": "抽選日"})
    df = df.sort_values("抽選日", ascending=False)
    return df

# ===== 特徴量作成 =====
def create_features(df, n_lags=10):
    feature_list = []
    label_list = []

    for i in range(n_lags, len(df)):
        window = df.iloc[i-n_lags:i]
        features = []
        for col in ["数字1", "数字2", "数字3", "数字4", "数字5", "数字6", "数字7", "数字8"]:
            counts = window[col].value_counts().reindex(range(1, 41), fill_value=0)
            features.extend(counts.values.tolist())
        feature_list.append(features)

        # ラベルはその回の数字（8個）
        row = df.iloc[i]
        labels = row[["数字1", "数字2", "数字3", "数字4", "数字5", "数字6", "数字7", "数字8"]].values.tolist()
        label_list.append(labels)

    return np.array(feature_list), np.array(label_list)

# ===== AI予測ロジック（学習済みモデルを使う） =====
def predict_numbers_by_ai(df, model_path="model/bingo5_model.pkl"):
    try:
        model = joblib.load(model_path)
    except FileNotFoundError:
        return [["学習済みモデルが見つかりません。"] * 5]

    X, _ = create_features(df)
    if len(X) == 0:
        return [["十分なデータがありません"] * 5]

    latest = X[-1].reshape(1, -1)
    results = []
    for _ in range(5):
        probs = model.predict_proba(latest)
        probs = np.array([p[:, 1] for p in probs])  # 各ラベルの確率
        top = np.argsort(-probs[0])[:8]
        numbers = sorted([i+1 for i in top])
        results.append([int(n) for n in numbers])
    return results

# ===== Streamlit UI =====
st.title("🎯 ビンゴ5出現数字おすすめジェネレーター")

st.markdown("""
📌 **推奨数字の生成ロジックを選んでください：**
""")

option = st.selectbox("", ["AI予測（学習モデル活用）"])

if st.button("📋 おすすめ数字を5口生成"):
    df = load_data()
    st.subheader("🎯 おすすめ数字（5口）")

    if option == "AI予測（学習モデル活用）":
        results = predict_numbers_by_ai(df)
        for i, r in enumerate(results):
            if isinstance(r, list):
                st.write(f"👉 {i+1}口目: {r}")
            else:
                st.error(f"AI予測時にエラーが発生しました: {r}")
