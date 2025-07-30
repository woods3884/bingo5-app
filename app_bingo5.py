import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import os
import joblib
from collections import Counter

# --- モデルの読み込み ---
@st.cache_resource
def load_model():
    model_path = "model/bingo5_model.pkl"
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

# --- データの読み込み ---
@st.cache_data
def load_data():
    df = pd.read_csv("data/date_bingo5.csv")
    df = df.rename(columns=lambda x: x.strip())
    df = df.sort_values("抽せん日", ascending=False)
    return df

# --- 頻出数字 ---
def show_frequent_numbers(df):
    nums = df[[f"数字{i}" for i in range(1, 9)]].values.flatten()
    counter = Counter(nums)
    most_common = counter.most_common()

    st.subheader("📈 頻出数字ランキング")
    fig, ax = plt.subplots()
    ax.bar([num for num, _ in most_common], [count for _, count in most_common])
    ax.set_xlabel("数字")
    ax.set_ylabel("出現回数")
    st.pyplot(fig)

# --- 未出数字 ---
def show_unshown_numbers(df):
    all_numbers = set(range(1, 41))
    appeared = set(df[[f"数字{i}" for i in range(1, 9)]].values.flatten())
    unshown = sorted(list(all_numbers - appeared))
    st.subheader("⚫ 未出数字")
    st.write(unshown)

# --- 連番傾向 ---
def show_consecutive_pattern(df):
    st.subheader("🔢 連番傾向")
    count = 0
    for _, row in df.iterrows():
        nums = sorted(row[[f"数字{i}" for i in range(1, 9)]].values)
        for i in range(len(nums) - 1):
            if nums[i] + 1 == nums[i + 1]:
                count += 1
                break
    st.write(f"連番が含まれる回数: {count} / {len(df)}")

# --- AI予測 ---
def show_ai_predictions(df, model):
    st.subheader("🤖 おすすめ数字（5口）")

    try:
        X = []
        for _, row in df.iterrows():
            nums = row[[f"数字{i}" for i in range(1, 9)]].values
            features = []
            features.append(np.mean(nums))
            features.append(np.std(nums))
            features.append(sum(n % 2 == 0 for n in nums))  # 偶数の数
            features.append(sum(n % 2 != 0 for n in nums))  # 奇数の数
            features.extend(nums)
            X.append(features)

        X = np.array(X)
        latest_X = X[:10]  # 最新10件分で生成

        for i in range(5):
            preds = model.predict(latest_X)
            pred_numbers = list(sorted(set(preds[i % len(preds)])))[:8]
            st.write(f"👉 {i+1}口目: {pred_numbers}")

    except Exception as e:
        st.error(f"AI予測時にエラーが発生しました: {e}")

# --- メインアプリ ---
st.title(" 🎯 ビンゴ5出現数字おすすめジェネレーター")

logic_option = st.selectbox("🕹️ 推奨数字の生成ロジックを選んでください：", [
    "頻出数字",
    "未出数字",
    "連番傾向",
    "AI予測（学習モデル活用）"
])

st.markdown("---")

model = load_model()
df = load_data()

if logic_option == "頻出数字":
    show_frequent_numbers(df)
elif logic_option == "未出数字":
    show_unshown_numbers(df)
elif logic_option == "連番傾向":
    show_consecutive_pattern(df)
elif logic_option == "AI予測（学習モデル活用）":
    if model is not None:
        show_ai_predictions(df, model)
    else:
        st.error("学習済みモデルが見つかりませんでした。model/bingo5_model.pkl を確認してください。")
