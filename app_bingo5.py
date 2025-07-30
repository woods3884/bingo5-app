# app_bingo5.py
import streamlit as st
import pandas as pd
import os
import random
import pickle
from collections import Counter
from sklearn.preprocessing import MultiLabelBinarizer

# --- タイトル ---
st.title("🎯 ビンゴ5出現数字おすすめジェネレーター")

# --- CSV データの読み込み ---
DATA_PATH = "data/date_bingo5.csv"
MODEL_PATH = "model/bingo5_model.pkl"

if not os.path.exists(DATA_PATH):
    st.error("CSVファイルが存在しません。")
    st.stop()

df = pd.read_csv(DATA_PATH)

# --- 過去の出現数字を取得 ---
all_numbers = df[[f"数字{i+1}" for i in range(8)]].values.flatten()
all_numbers = pd.Series(all_numbers).dropna().astype(int)

# --- 頻出・未出・連番 ---
freq_counter = Counter(all_numbers)
most_common = [num for num, _ in freq_counter.most_common()]
missing_numbers = [i for i in range(1, 41) if i not in all_numbers.values]

# --- 連番傾向 ---
consecutive_count = 0
for row in df[[f"数字{i+1}" for i in range(8)]].values:
    nums = sorted([int(n) for n in row if pd.notna(n)])
    for i in range(len(nums) - 1):
        if nums[i+1] - nums[i] == 1:
            consecutive_count += 1
            break

# --- 推奨ロジック選択 ---
logic = st.selectbox("🧠 推奨数字の生成ロジックを選んでください：", [
    "頻出数字ベース",
    "未出数字ベース",
    "連番重視ベース",
    "AI風バランス生成（頻繁出 + ランダム）",
    "AI予測（学習モデル活用）"
])

# --- 推奨生成関数 ---
def generate_numbers(logic):
    if logic == "頻出数字ベース":
        return sorted(random.sample(most_common[:20], 8))
    
    elif logic == "未出数字ベース":
        if len(missing_numbers) >= 8:
            return sorted(random.sample(missing_numbers, 8))
        else:
            others = [i for i in range(1, 41) if i not in missing_numbers]
            fill = random.sample(others, 8 - len(missing_numbers))
            return sorted(missing_numbers + fill)

    elif logic == "連番重視ベース":
        base = random.randint(1, 39)
        nums = [base, base+1]
        remain = [i for i in range(1, 41) if i not in nums]
        nums += random.sample(remain, 6)
        return sorted(nums)

    elif logic == "AI風バランス生成（頻繁出 + ランダム）":
        base = random.sample(most_common[:25], 4)
        remain = [i for i in range(1, 41) if i not in base]
        base += random.sample(remain, 4)
        return sorted(base)

    elif logic == "AI予測（学習モデル活用）":
        if not os.path.exists(MODEL_PATH):
            st.warning("学習済みモデルが見つかりません。")
            return []

        try:
            with open(MODEL_PATH, "rb") as f:
                model = pickle.load(f)
            
            latest_draws = df[[f"数字{i+1}" for i in range(8)]].tail(10).fillna(0).astype(int).values.tolist()
            mlb = MultiLabelBinarizer(classes=list(range(1, 41)))
            X_pred = mlb.fit_transform(latest_draws)

            y_preds = model.predict_proba(X_pred)
            avg_probs = sum(y_preds) / len(y_preds)

            top8 = sorted(sorted(range(1, 41), key=lambda i: -avg_probs[i - 1])[:8])
            return top8
        except Exception as e:
            st.error(f"AI予測時にエラーが発生しました: {e}")
            return []

# --- ボタン押下で生成 ---
if st.button("🔁 おすすめ数字を5口生成"):
    st.subheader("🎯 おすすめ数字（5口）")
    for i in range(5):
        nums = generate_numbers(logic)
        st.markdown(f"👉 **{i+1}口目:** {nums}")
