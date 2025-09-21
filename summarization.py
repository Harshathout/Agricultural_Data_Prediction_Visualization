import streamlit as st
import pandas as pd
from transformers import pipeline

st.set_page_config(page_title="Text Summarization", layout="wide")
st.title("Mini AI: Text Summarization for Agriculture Data")

# ------------------- Load summarizer -------------------
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

st.info("Loading AI summarizer model... this may take a few seconds.")
summarizer = load_summarizer()

# ------------------- Upload dataset -------------------
uploaded_file = st.file_uploader("Upload CSV/Excel dataset to summarize", type=['csv', 'xls', 'xlsx'])

if uploaded_file:
    try:
        # Load Excel or CSV
        if uploaded_file.name.endswith('.xls'):
            df = pd.read_excel(uploaded_file, engine='xlrd')
        elif uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file, engine='openpyxl')
        else:
            try:
                df = pd.read_csv(uploaded_file, encoding='utf-8')
            except UnicodeDecodeError:
                df = pd.read_csv(uploaded_file, encoding='latin1')

        # Fill missing values
        df_clean = df.fillna(method='ffill')
        st.subheader("Cleaned Dataset")
        st.dataframe(df_clean)

        # ------------------- Summarization -------------------
        text_column = st.text_input("Enter the column name to summarize:")

        if st.button("Generate Summaries"):
            if text_column in df_clean.columns:
                st.write("Summarizing text... this may take a few seconds.")
                progress = st.progress(0)
                summaries = []

                for i, text in enumerate(df_clean[text_column].astype(str).tolist()):
                    if text.strip():
                        summary = summarizer(text, max_length=50, min_length=20, do_sample=False)
                        summaries.append(summary[0]['summary_text'])
                    else:
                        summaries.append("")
                    progress.progress((i + 1)/len(df_clean))

                df_clean[f"{text_column}_summary"] = summaries
                st.subheader("Summarized Column")
                st.dataframe(df_clean[[text_column, f"{text_column}_summary"]])

                # ------------------- Download Button -------------------
                st.download_button(
                    label="Download Summarized Dataset",
                    data=df_clean.to_csv(index=False).encode('utf-8'),
                    file_name="summarized_data.csv",
                    mime="text/csv"
                )
            else:
                st.warning("Column not found in dataset.")

    except Exception as e:
        st.error(f"Error reading file: {e}")
