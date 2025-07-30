# app_bingo5.py
import streamlit as st
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import os

# CSVファイル読み込み
DATA_PATH = "data/date_bingo5.csv"

st.title("🎯 ビンゴ5 出現数字ランキング")

if not os.path.exists(DATA_PATH):
    st.error("CSVファイルが存在しません。")
else:
    df = pd.read_csv(DATA_PATH)

    st.subheader("📅 最新データ表示")
    st.dataframe(df.tail(10))

    # 全出現数字を集計
    all_numbers = df[[f"数字{i+1}" for i in range(8)]].values.flatten()
    counter = Counter(all_numbers)
    most_common = counter.most_common()

    st.subheader("📊 頻出数字ランキング")
    result_df = pd.DataFrame(most_common, columns=["数字", "出現回数"])
    st.dataframe(result_df)

    # グラフ
    st.subheader("📈 出現回数グラフ")
    fig, ax = plt.subplots()
    ax.bar(result_df["数字"], result_df["出現回数"])
    ax.set_xlabel("数字")
    ax.set_ylabel("出現回数")
    st.pyplot(fig)
