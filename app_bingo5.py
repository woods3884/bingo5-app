# app_bingo5.py
import streamlit as st
import pandas as pd
import random
import os
from collections import Counter

st.title("🎯 ビンゴ5出現数字おすすめジェネレーター")

DATA_PATH = "data/date_bingo5.csv"

if not os.path.exists(DATA_PATH):
    st.error("CSVファイルが存在しません。")
else:
    df = pd.read_csv(DATA_PATH)

    all_drawn_numbers = df[[f"数字{i+1}" for i in range(8)]].values.flatten()
    all_drawn_numbers = pd.Series(all_drawn_numbers).dropna().astype(int)

    freq_counter = Counter(all_drawn_numbers)
    most_common = [num for num, _ in freq_counter.most_common()]
    missing_numbers = [i for i in range(1, 41) if i not in all_drawn_numbers.values]

    logic = st.selectbox("🧠 推奨数字の生成ロジックを選んでください：", [
        "頻出数字ベース",
        "未出数字ベース",
        "連番重視ベース",
        "AI風バランス生成（頻出＋ランダム）"
    ])

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
            nums = []
            base = random.randint(1, 39)
            nums += [base, base + 1]
            remain = [i for i in range(1, 41) if i not in nums]
            nums += random.sample(remain, 6)
            return sorted(nums)
        elif logic == "AI風バランス生成（頻出＋ランダム）":
            base_nums = most_common[:15]
            rand_nums = [i for i in range(1, 41) if i not in base_nums]
            selected = random.sample(base_nums, 5) + random.sample(rand_nums, 3)
            return sorted(selected)

    if st.button("🔁 おすすめ数字を5口生成"):
        st.subheader("🎯 おすすめ数字（5口）")
        for i in range(5):
            recommendation = generate_numbers(logic)
            st.write(f"👉 {i+1}口目: **{recommendation}**")
