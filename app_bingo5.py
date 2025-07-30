# app_bingo5.py
import streamlit as st
import pandas as pd
import random
import os
from collections import Counter

st.title("🎯 ビンゴ5 出現数字おすすめジェネレーター")

DATA_PATH = "data/date_bingo5.csv"

if not os.path.exists(DATA_PATH):
    st.error("CSVファイルが存在しません。")
else:
    df = pd.read_csv(DATA_PATH)

    # 1. 全出現数字をリスト化
    all_drawn_numbers = df[[f"数字{i+1}" for i in range(8)]].values.flatten()
    all_drawn_numbers = pd.Series(all_drawn_numbers).dropna().astype(int)

    # 2. 頻出ランキング
    freq_counter = Counter(all_drawn_numbers)
    most_common = [num for num, _ in freq_counter.most_common()]

    # 3. 未出数字（1〜40で1度も出てない）
    missing_numbers = [i for i in range(1, 41) if i not in all_drawn_numbers.values]

    # 4. 連番傾向：各行で連番が含まれている場合カウント
    consecutive_count = 0
    for row in df[[f"数字{i+1}" for i in range(8)]].values:
        nums = sorted([int(n) for n in row if pd.notna(n)])
        for i in range(len(nums) - 1):
            if nums[i+1] - nums[i] == 1:
                consecutive_count += 1
                break

    # 5. セレクトボックス
    logic = st.selectbox("🧠 おすすめ数字の生成ロジックを選んでください：", [
        "頻出数字ベース",
        "未出数字ベース",
        "連番重視ベース"
    ])

    def generate_recommendation(logic):
        if logic == "頻出数字ベース":
            return sorted(random.sample(most_common[:20], 8))
        elif logic == "未出数字ベース":
            if len(missing_numbers) >= 8:
                return sorted(random.sample(missing_numbers, 8))
            else:
                # 未出が少ないときは残りをランダム補充
                others = [i for i in range(1, 41) if i not in missing_numbers]
                fill = random.sample(others, 8 - len(missing_numbers))
                return sorted(missing_numbers + fill)
        elif logic == "連番重視ベース":
            # 連番を最低1組含むように設計
            nums = []
            base = random.randint(1, 39)
            nums += [base, base+1]
            remain = [i for i in range(1, 41) if i not in nums]
            nums += random.sample(remain, 6)
            return sorted(nums)

    if st.button("🔁 おすすめ数字を生成"):
        recommendation = generate_recommendation(logic)
        st.success(f"🎉 おすすめ数字: {recommendation}")
