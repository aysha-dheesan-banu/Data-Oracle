import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import io
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder

# 🛠️ Streamlit App Configuration
st.set_page_config(page_title="🔮 Data Oracle – AI Insight Companion", layout="wide")

st.markdown("""
    <style>
        .main {background-color: #f0f8ff;}
        .stButton>button {
            background-color: #4b0082;
            color: white;
            font-weight: bold;
            border-radius: 8px;
        }
        .css-18e3th9 {
            background-color: #e6f2ff;
        }
        .reportview-container .markdown-text-container {
            font-family: 'Courier New';
            font-size: 16px;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🔮 Data Oracle – Your AI Companion for Smart Insights")
st.caption("✨ Powered by AI magic. Upload your data. Let the Oracle speak.")

# 📥 Upload Your Data
uploaded_file = st.file_uploader("📂 Drop your sacred CSV scroll here", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.balloons()
    st.success("🎉 Oracle received your data. Insights are brewing...")

    # 🧭 Navigation
    st.sidebar.title("🧭 Oracle Navigation Portal")
    section = st.sidebar.radio("Choose your wisdom path", [
        "📖 Scroll Overview",
        "🧩 Missing Mysteries",
        "📊 Visual Prophecy",
        "🧪 Outlier Oracle",
        "🎨 Insight Alchemist",
        "🧠 Model Muse",
        "⚙️ AI Armory",
        "📜 Oracle’s Scroll"
    ])

    # 🔣 Label Encoding Helper
    def label_encode(df):
        le = LabelEncoder()
        for col in df.select_dtypes(include='object').columns:
            df[col] = le.fit_transform(df[col])
        return df

    # 🔍 Isolation Forest Outlier Detection
    def find_oracles_outliers(df):
        model = IsolationForest(contamination=0.05)
        preds = model.fit_predict(df.select_dtypes(include='number'))
        df['Oracle_Outlier'] = preds
        return df[df['Oracle_Outlier'] == -1]

    # 📜 Oracle Report Generator
    def generate_scroll(df):
        scroll = io.StringIO()
        scroll.write("🔮 ORACLE SCROLL OF DATA WISDOM 🔮\n")
        scroll.write("=" * 60 + "\n\n")
        scroll.write("✨ Basic Essence of the Dataset:\n")
        scroll.write(f"📏 Shape: {df.shape}\n")
        scroll.write(f"🔢 Numeric Fields: {list(df.select_dtypes(include='number').columns)}\n")
        scroll.write(f"🔤 Categorical Fields: {list(df.select_dtypes(include='object').columns)}\n\n")

        scroll.write("❗ Missing Value Ritual:\n")
        missing = df.isnull().sum()
        missing = missing[missing > 0]
        scroll.write(missing.to_string() + "\n\n" if not missing.empty else "None! The dataset is whole.\n\n")

        scroll.write("🔍 Outlier Vision (IQR):\n")
        for col in df.select_dtypes(include='number').columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)]
            scroll.write(f"🔺 {col}: {len(outliers)} strange signals\n")
        scroll.write("\n")

        scroll.write("📉 Statistical Incantations:\n")
        scroll.write(df.describe().to_string())
        scroll.write("\n\n")

        scroll.write("🧲 Correlation Aura:\n")
        scroll.write(df.select_dtypes(include='number').corr().to_string())
        scroll.write("\n\n")

        scroll.write("🧠 Model Guidance:\n")
        if len(df.select_dtypes(include='number').columns) >= 3:
            scroll.write("✔️ Regression / Clustering viable.\n")
        if len(df.select_dtypes(include='object').columns) >= 1:
            scroll.write("✔️ Classification spells applicable.\n")
        scroll.write("\n✨ Oracle’s Advice:\n")
        scroll.write("- Handle null voids 🕳️.\n- Banish outliers 👻.\n- Cast encodings 🪄.\n- Explore visual spells 🖼️.\n")

        return scroll.getvalue()

    # 🌟 Oracle Navigation Sections
    if section == "📖 Scroll Overview":
        st.header("📖 First Glance: The Sacred Scroll")
        st.dataframe(df.head())

    elif section == "🧩 Missing Mysteries":
        st.header("🧩 Uncovering the Voids")
        missing = df.isnull().sum()
        st.dataframe(missing[missing > 0].reset_index().rename(columns={'index': 'Field', 0: 'Empty Realms'}))

    elif section == "📊 Visual Prophecy":
        st.header("📊 Charts of Destiny")
        for col in df.select_dtypes(include='number').columns[:3]:
            st.plotly_chart(px.histogram(df, x=col, title=f"🔮 Distribution of {col}"))

    elif section == "🧪 Outlier Oracle":
        st.header("🧪 Outliers from the Shadows")
        for col in df.select_dtypes(include='number').columns[:3]:
            st.plotly_chart(px.box(df, y=col, title=f"🧊 {col} Box Vision"))
        
        st.subheader("✨ Oracle’s Isolation Forest Ritual")
        outliers = find_oracles_outliers(df)
        st.dataframe(outliers)

    elif section == "🎨 Insight Alchemist":
        st.header("🎨 Insight Alchemy Chamber")
        x = st.selectbox("🔹 X-Axis", df.columns)
        y = st.selectbox("🔸 Y-Axis", df.columns)
        chart_type = st.selectbox("Choose your visual spell:", ["Scatter", "Bar", "Line", "Box"])
        if st.button("🔮 Create Insight"):
            if chart_type == "Scatter":
                st.plotly_chart(px.scatter(df, x=x, y=y))
            elif chart_type == "Bar":
                st.plotly_chart(px.bar(df, x=x, y=y))
            elif chart_type == "Line":
                st.plotly_chart(px.line(df, x=x, y=y))
            elif chart_type == "Box":
                st.plotly_chart(px.box(df, x=x, y=y))

    elif section == "🧠 Model Muse":
        st.header("🧠 Model Muse Whispers")
        st.markdown("✨ The data reveals the path...")
        if len(df.select_dtypes(include='number').columns) >= 3:
            st.success("🔁 Consider Regression & Clustering.")
        if len(df.select_dtypes(include='object').columns) >= 1:
            st.success("🧠 Classification models are open to you.")

    elif section == "⚙️ AI Armory":
        st.header("⚙️ Advanced Oracle Tools")
        st.markdown("🔮 Prepare to summon:")
        st.markdown("- 🌲 Random Forest for structured complexity")
        st.markdown("- ⚡ XGBoost for precision spells")
        st.markdown("- 🌀 KMeans for unsupervised separation of entities")

    elif section == "📜 Oracle’s Scroll":
        st.header("📜 Wisdom Scroll Report")
        scroll_text = generate_scroll(df)
        st.text_area("Read the Oracle’s scroll:", scroll_text, height=400)
        st.download_button("📥 Download Oracle’s Scroll", scroll_text, file_name="oracle_report.txt")

else:
    st.info("🧙 Awaiting your scroll (CSV). Drop it to awaken the Oracle.")
