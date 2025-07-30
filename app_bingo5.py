import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# --- データ読み込み ---
@st.cache_data
def load_data():
    df = pd.read_csv("data/date_bingo5.csv")
    df = df.rename(columns=lambda x: x.strip())
    df = df.sort_values("抽せん日", ascending=False)
    return df

# --- 特徴量作成 ---
def create_features(df):
    df_feat = df.copy()
    for i in range(1, 9):
        df_feat[f"num{i}"] = df_feat[f"数字{i}"]
    for i in range(1, 41):
        df_feat[f"feature_{i}"] = df_feat[[f"num{j}" for j in range(1, 9)]].apply(
            lambda row: int(i in row.values), axis=1
        )
    return df_feat

# --- AI予測（修正版） ---
def predict_numbers_by_ai(df):
    latest = df.iloc[[-1]]
    feature_cols = [col for col in df.columns if col.startswith('feature_')]
    latest_features = latest[feature_cols].values  # ndarray に変換
    model = joblib.load("model/bingo5_model.pkl")

    probs = model.predict_proba(latest_features).reshape(-1)  # shape (40,)
    top8 = np.argsort(probs)[::-1][:8]
    result = sorted([int(n + 1) for n in top8])  # 0-index → 1-index
    return result

# --- 頻出数字取得 ---
def get_frequent_numbers(df):
    numbers = df[[f"数字{i}" for i in range(1, 9)]].values.flatten()
    return Counter(numbers)

# --- Streamlit UI ---
st.title("🎯 ビンゴ5出現数字おすすめジェネレーター")

logic = st.selectbox(
    "🧊 推奨数字の生成ロジックを選んでください：",
    ["頻出数字", "未出数字", "ランダム", "AI予測（学習モデル活用）"]
)

if st.button("📋 おすすめ数字を5口生成"):
    st.markdown("### 🎯 おすすめ数字（5口）")
    df_raw = load_data()
    df_feat = create_features(df_raw)

    for i in range(5):
        try:
            if logic == "頻出数字":
                freq = get_frequent_numbers(df_raw)
                top8 = [num for num, _ in freq.most_common(8)]
                result = sorted(np.random.choice(top8, 8, replace=False).tolist())

            elif logic == "未出数字":
                all_nums = set(range(1, 41))
                used_nums = set(df_raw[[f"数字{i}" for i in range(1, 9)]].values.flatten())
                unused = list(all_nums - used_nums)
                if len(unused) < 8:
                    unused += list(all_nums)
                result = sorted(np.random.choice(unused, 8, replace=False).tolist())

            elif logic == "ランダム":
                result = sorted(np.random.choice(range(1, 41), 8, replace=False).tolist())

            elif logic == "AI予測（学習モデル活用）":
                result = predict_numbers_by_ai(df_feat)

            st.write(f"👉 {i+1}口目: {result}")

        except Exception as e:
            st.error(f"AI予測時にエラーが発生しました: {e}")

# --- 頻出数字の可視化 ---
if logic == "頻出数字":
    st.markdown("## 🔢 頻出数字ランキング")
    df = load_data()
    freq = get_frequent_numbers(df)
    freq_df = pd.DataFrame(freq.items(), columns=["数字", "出現回数"]).sort_values("数字")
    plt.figure(figsize=(10, 4))
    sns.barplot(x="数字", y="出現回数", data=freq_df, color="skyblue")
    plt.xticks(rotation=90)
    st.pyplot(plt)
