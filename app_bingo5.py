# app_bingo5.py
import streamlit as st
import pandas as pd
import random
import os
import pickle
from collections import Counter

from sklearn.preprocessing import MultiLabelBinarizer

st.title("🎯 ビンゴ5出現数字おすすめジェネレーター")

DATA_PATH = "data/date_bingo5.csv"
MODEL_PATH = "bingo5_model.pkl"

if not os.path.exists(DATA_PATH):
    st.error("CSVファイルが存在しません。")
else:
    df = pd.read_csv(DATA_PATH)

    all_drawn_numbers = df[[f"数字{i+1}" for i in range(8)]].values.flatten()
    all_drawn_numbers = pd.Series(all_drawn_numbers).dropna().astype(int)

    freq_counter = Counter(all_drawn_numbers)
    most_common = [num for num, _ in freq_counter.most_common()]
    missing_numbers = [i for i in range(1, 41) if i not in all_drawn_numbers.values]

    consecutive_count = 0
    for row in df[[f"数字{i+1}" for i in range(8)]].values:
        nums = sorted([int(n) for n in row if pd.notna(n)])
        for i in range(len(nums) - 1):
            if nums[i+1] - nums[i] == 1:
                consecutive_count += 1
                break

    logic = st.selectbox("🧠 おすすめ数字の生成ロジックを選んでください：", [
        "頻出数字ベース",
        "未出数字ベース",
        "連番重視ベース",
        "AI風バランス生成（頻繁出 + ランダム）",
        "AI予測（学習モデル使用）"
    ])

    def generate_recommendation(logic):
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
            nums = []
            base = random.randint(1, 39)
            nums += [base, base+1]
            remain = [i for i in range(1, 41) if i not in nums]
            nums += random.sample(remain, 6)
            return sorted(nums)

        elif logic == "AI風バランス生成（頻繁出 + ランダム）":
            nums = random.sample(most_common[:25], 5) + random.sample(range(1, 41), 3)
            return sorted(set(nums))[:8]

        elif logic == "AI予測（学習モデル使用）":
            if not os.path.exists(MODEL_PATH):
                st.warning("学習済みモデルが見つかりません。")
                return []
            with open(MODEL_PATH, "rb") as f:
                model = pickle.load(f)

            # 最新10件の出現数字を特徴量とする例（シンプルな実装）
            recent = df[[f"数字{i+1}" for i in range(8)]].tail(10).values
            recent_numbers = [int(n) for row in recent for n in row if pd.notna(n)]
            mlb = MultiLabelBinarizer(classes=range(1, 41))
            recent_encoded = mlb.fit_transform([recent_numbers])
            preds = model.predict_proba(recent_encoded)[0]
            top_predicted = sorted(range(1, 41), key=lambda i: preds[i - 1], reverse=True)
            return sorted(top_predicted[:8])

    if st.button("🔁 おすすめ数字を5口生成"):
        st.subheader("🎯 おすすめ数字（5口）")
        for i in range(5):
            numbers = generate_recommendation(logic)
            st.write(f"👉 {i+1}口目: {numbers}")
