import streamlit as st
import pandas as pd
import numpy as np
import random
import os
import pickle
from sklearn.preprocessing import MultiLabelBinarizer

# --- データ読み込み ---
DATA_PATH = "data/date_bingo5.csv"
MODEL_PATH = "model/bingo5_model.pkl"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df.sort_values("抽選日", ascending=False)
    df.reset_index(drop=True, inplace=True)
    return df

def load_model():
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    else:
        return None

# --- 数字生成ロジック ---
def generate_random_numbers():
    return sorted(random.sample(range(1, 40), 8))

def generate_from_freq(df):
    all_nums = df[[f"num{i}" for i in range(1, 9)]].values.flatten()
    freq = pd.Series(all_nums).value_counts().sort_values(ascending=False)
    top20 = freq.head(20).index.tolist()
    return sorted(random.sample(top20, 8))

def generate_from_unseen(df):
    all_history = df[[f"num{i}" for i in range(1, 9)]].values.flatten()
    unseen = [n for n in range(1, 40) if n not in all_history]
    pool = unseen if unseen else list(range(1, 40))
    return sorted(random.sample(pool, 8))

def generate_ai_prediction(model, df):
    try:
        latest = df.sort_values('抽選日', ascending=False).head(4)
        latest_numbers = latest[[f'num{i}' for i in range(1, 9)]].values.flatten()
        features = [f'num{i}_t{j}' for j in range(1, 5) for i in range(1, 9)]
        X_input = pd.DataFrame([latest_numbers], columns=features)
        y_pred = model.predict(X_input)
        mlb = MultiLabelBinarizer(classes=list(range(1, 40)))
        mlb.fit([[]])
        y_decoded = mlb.inverse_transform(y_pred)
        return sorted(list(y_decoded[0]))
    except Exception as e:
        return f"AI予測時にエラーが発生しました: {e}"

# --- Streamlit UI ---
st.title("🎯 ビンゴ5出現数字おすすめジェネレーター")

st.markdown("""
- 推奨数字の生成ロジックを選んでください：
""")

logic = st.selectbox("",
                     ["ランダム生成（完全ランダム）",
                      "頻出数字ベース（過去データ）",
                      "未出数字ベース（過去データ）",
                      "AI予測（学習モデル活用）"])

st.markdown("---")

if st.button("📋 おすすめ数字を5口生成"):
    df = load_data()
    model = load_model() if logic == "AI予測（学習モデル活用）" else None

    st.subheader("🎯 おすすめ数字（5口）")
    for i in range(5):
        if logic == "ランダム生成（完全ランダム）":
            nums = generate_random_numbers()
        elif logic == "頻出数字ベース（過去データ）":
            nums = generate_from_freq(df)
        elif logic == "未出数字ベース（過去データ）":
            nums = generate_from_unseen(df)
        elif logic == "AI予測（学習モデル活用)":
            nums = generate_ai_prediction(model, df)
        else:
            nums = []

        if isinstance(nums, str):
            st.error(nums)
        else:
            st.write(f"👉 {i+1}口目: {nums}")
