import streamlit as st
import pandas as pd
from transformers import pipeline

st.title("Text Categorization")

# ------------------- Load classifier -------------------
@st.cache_resource
def load_classifier():
    return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

st.info("Loading AI model... this may take a few seconds.")
classifier = load_classifier()

# ------------------- Upload dataset -------------------
uploaded_file = st.file_uploader("Upload CSV/Excel dataset to categorize", type=['csv', 'xls', 'xlsx'])

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

        df_clean = df.fillna(method='ffill')
        st.subheader("Cleaned Dataset")
        st.dataframe(df_clean)

        # ------------------- Categorization -------------------
        cat_column = st.text_input("Enter the column name to categorize:")
        candidate_labels = st.text_area("Enter categories (comma separated, e.g., 'Area, Production, Yield'):")

        if st.button("Generate Categories"):
            if cat_column in df_clean.columns and candidate_labels:
                labels = [x.strip() for x in candidate_labels.split(",")]
                st.write("Categorizing text... this may take a few seconds.")
                progress = st.progress(0)
                categories = []

                for i, text in enumerate(df_clean[cat_column].astype(str).tolist()):
                    if text.strip():
                        result = classifier(text, candidate_labels=labels)
                        categories.append(result['labels'][0])
                    else:
                        categories.append("")
                    progress.progress((i+1)/len(df_clean))

                df_clean[f"{cat_column}_category"] = categories
                st.subheader("Categorized Column")
                st.dataframe(df_clean[[cat_column, f"{cat_column}_category"]])

                # Download button
                st.download_button(
                    label="Download Categorized Dataset",
                    data=df_clean.to_csv(index=False).encode('utf-8'),
                    file_name="categorized_data.csv",
                    mime="text/csv"
                )
            else:
                st.warning("Provide a valid column name and candidate categories.")

    except Exception as e:
        st.error(f"Error reading file: {e}")
