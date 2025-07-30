import streamlit as st
import pandas as pd
import numpy as np
import random
import os
import joblib
from sklearn.preprocessing import MultiLabelBinarizer

# CSV 読み込み関数（列名変換を含む）
@st.cache_data
def load_data():
    df = pd.read_csv("data/date_bingo5.csv")
    df.rename(columns={"抽せん日": "抽選日"}, inplace=True)  # 列名変換
    df = df.sort_values("抽選日", ascending=False)
    return df

# 出現頻度をカウント
def get_frequency(df):
    nums = df[[f"数字{i}" for i in range(1, 9)]].values.flatten()
    return pd.Series(nums).value_counts().sort_index()

# 未出数字を抽出
def get_unseen_numbers(df):
    all_nums = set(range(1, 41))
    seen = set(df[[f"数字{i}" for i in range(1, 9)]].values.flatten())
    return sorted(all_nums - seen)

# 連番ペアを抽出（例: 12と13など）
def get_consecutive_pairs(df):
    pairs = []
    for row in df[[f"数字{i}" for i in range(1, 9)]].values:
        row = sorted(row)
        for i in range(len(row)-1):
            if row[i+1] - row[i] == 1:
                pairs.append((row[i], row[i+1]))
    return pairs

# AIモデルで予測（特徴量40次元）
def predict_numbers_by_ai(model, df, n_sets=5):
    results = []
    feature_columns = [f"数字{i}" for i in range(1, 9)]
    recent_draws = df[feature_columns].head(10).values

    for _ in range(n_sets):
        try:
            x_input = recent_draws.flatten()
            if len(x_input) != 80:
                raise ValueError("特徴量数が80個ではありません（8列×10行）")

            # 特徴量として上位40個の出現数を使用
            freq = pd.Series(x_input).value_counts().sort_index()
            all_features = [freq.get(i, 0) for i in range(1, 41)]

            probs = model.predict_proba([all_features])
            if isinstance(probs, list):
                probs = np.array([p[:, 1] for p in probs]).T

            top = np.argsort(-probs[0])[:8]
            numbers = sorted([i+1 for i in top])
            results.append(numbers)

        except Exception as e:
            st.warning(f"AI予測時にエラーが発生しました: {e}")
            results.append([])

    return results

# Streamlit UI
st.title("🎯 ビンゴ5出現数字おすすめジェネレーター")

logic = st.selectbox("🎲 推奨数字の生成ロジックを選んでください：", ["頻出数字", "未出数字", "連番傾向", "AI予測（学習モデル活用）"])

if logic == "AI予測（学習モデル活用）":
    if st.button("📄 おすすめ数字を5口生成"):
        df = load_data()
        model_path = "model/bingo5_model.pkl"
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            predicted = predict_numbers_by_ai(model, df, n_sets=5)
            st.markdown("## 🎯 おすすめ数字（5口）")
            for i, nums in enumerate(predicted, 1):
                st.write(f"👉 {i}口目: {nums}")
        else:
            st.warning("学習済みモデルが見つかりません。")

elif logic == "頻出数字":
    df = load_data()
    st.markdown("## 🔢 頻出数字ランキング")
    freq = get_frequency(df)
    st.bar_chart(freq)

elif logic == "未出数字":
    df = load_data()
    unseen = get_unseen_numbers(df)
    st.markdown("## ❌ 未出数字一覧")
    st.write(unseen)

elif logic == "連番傾向":
    df = load_data()
    pairs = get_consecutive_pairs(df)
    st.markdown("## 🔗 連番ペア出現履歴")
    st.write(pairs)
