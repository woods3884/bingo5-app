import streamlit as st
import pandas as pd
import random
import os
import joblib
from collections import Counter

st.title("\U0001F3AF ビンゴ5出現数字おすすめジェネレーター")

DATA_PATH = "data/date_bingo5.csv"
MODEL_PATH = "model/bingo5_model.pkl"

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

    # 4. セレクトボックスでロジック選択
    logic = st.selectbox("🧠 推奨数字の生成ロジックを選んでください：", [
        "頻出数字ベース",
        "未出数字ベース",
        "連番重視ベース",
        "AI予測（学習モデル活用）"
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

        elif logic == "AI予測（学習モデル活用）":
            return predict_with_model(df)

    def predict_with_model(df, n_predictions=5):
        try:
            if df.shape[0] < 10:
                st.warning("AI予測に必要な過去データが10件未満のため、予測できません。")
                return []

            if not os.path.exists(MODEL_PATH):
                st.warning("学習済みモデルが見つかりません。")
                return []

            model = joblib.load(MODEL_PATH)

            latest_data = df.iloc[-10:][[f"数字{i+1}" for i in range(8)]].values
            latest_features = latest_data.flatten().reshape(1, -1)

            y_pred = model.predict(latest_features)
            pred_indices = list(y_pred[0].argsort()[-8:][::-1])
            return sorted([i+1 for i in pred_indices])

        except Exception as e:
            st.error(f"AI予測時にエラーが発生しました: {e}")
            return []

    if st.button("\U0001F4BE おすすめ数字を5口生成"):
        st.subheader("\U0001F3AF おすすめ数字（5口）")
        for i in range(5):
            numbers = generate_recommendation(logic)
            st.write(f"👉 {i+1}口目: {numbers}")
