import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from transformers import pipeline

st.set_page_config(page_title="Agriculture AI Dashboard", layout="wide")
st.title("Agricultural Data: Summarization, Categorization, Prediction & Visualization")


@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

@st.cache_resource
def load_classifier():
    return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

st.info("Loading AI models... this may take a few seconds.")
summarizer = load_summarizer()
classifier = load_classifier()


uploaded_file = st.file_uploader("Upload CSV/Excel dataset", type=['csv', 'xls', 'xlsx'])

if uploaded_file:
    try:
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

        st.subheader("Data Visualizations (Cleaned Dataset)")
        required_cols = ['Crop', 'Production', 'Area', 'Yield ']
        if all(col in df_clean.columns for col in required_cols):

            st.write("Total Production per Crop")
            plt.figure(figsize=(10,5))
            sns.barplot(data=df_clean, x='Crop', y='Production', palette='viridis')
            plt.xticks(rotation=45)
            plt.grid(True)
            st.pyplot(plt.gcf())
            plt.clf()

            st.write("Yield per Crop")
            plt.figure(figsize=(10,5))
            sns.barplot(data=df_clean, x='Crop', y='Yield ', palette='magma')
            plt.xticks(rotation=45)
            plt.grid(True)
            st.pyplot(plt.gcf())
            plt.clf()

            st.write("Correlation Heatmap (Cleaned Data)")
            plt.figure(figsize=(8,6))
            sns.heatmap(df_clean[required_cols[1:]].corr(), annot=True, fmt=".2f", cmap="coolwarm")
            st.pyplot(plt.gcf())
            plt.clf()
        else:
            st.warning("Dataset must contain 'Crop', 'Area', 'Production', and 'Yield ' columns for visualization.")


        st.subheader("Text Summarization")
        text_column = st.text_input("Enter column name to summarize:")
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
                st.download_button(
                    label="Download Summarized Dataset",
                    data=df_clean.to_csv(index=False).encode('utf-8'),
                    file_name="summarized_data.csv",
                    mime="text/csv"
                )
            else:
                st.warning("Column not found in dataset.")


        st.subheader("Text Categorization (Zero-Shot)")
        cat_column = st.text_input("Enter column name for categorization:")
        candidate_labels = st.text_area("Enter candidate categories (comma separated, e.g., 'Area, Production, Yield'):")
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
                st.dataframe(df_clean[[cat_column, f"{cat_column}_category"]])
            else:
                st.warning("Column not found or candidate labels empty.")

        if all(col in df_clean.columns for col in required_cols):
            st.subheader("Predictions (Optional)")
            non_total_mask = df_clean['Crop'].str.contains("Total")
            X = df_clean.loc[non_total_mask, ['Area', 'Yield ']]
            y = df_clean.loc[non_total_mask, 'Production']
            model = LinearRegression()
            model.fit(X, y)
            df_clean.loc[non_total_mask, 'Production_Predicted'] = model.predict(X)
            df_clean.loc[non_total_mask, 'Prod_per_Area_Predicted'] = df_clean.loc[non_total_mask, 'Production_Predicted'] / df_clean.loc[non_total_mask, 'Area']
            df_clean.loc[non_total_mask, 'Yield_Predicted'] = df_clean.loc[non_total_mask, 'Production_Predicted'] / df_clean.loc[non_total_mask, 'Area']

            st.dataframe(df_clean)

            st.subheader("Data Visualizations (Predicted Values)")
            plt.figure(figsize=(10,5))
            sns.barplot(data=df_clean[non_total_mask], x='Crop', y='Yield_Predicted', palette='viridis')
            plt.xticks(rotation=45)
            plt.grid(True)
            st.pyplot(plt.gcf())
            plt.clf()

            plt.figure(figsize=(10,5))
            sns.barplot(data=df_clean[non_total_mask], x='Crop', y='Prod_per_Area_Predicted', palette='magma')
            plt.xticks(rotation=45)
            plt.grid(True)
            st.pyplot(plt.gcf())
            plt.clf()

            plt.figure(figsize=(8,6))
            numeric_cols = ['Area', 'Yield ', 'Production_Predicted', 'Prod_per_Area_Predicted', 'Yield_Predicted']
            sns.heatmap(df_clean.loc[non_total_mask, numeric_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm")
            st.pyplot(plt.gcf())
            plt.clf()

            st.download_button(
                label="Download Dataset with Predictions",
                data=df_clean.to_csv(index=False).encode('utf-8'),
                file_name="agriculture_ai_data.csv",
                mime="text/csv"
            )

    except Exception as e:
        st.error(f"Error reading file: {e}")
