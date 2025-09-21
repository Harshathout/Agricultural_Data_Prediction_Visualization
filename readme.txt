# Rural Development Agricultural Data Analysis

## Overview

This project involves collecting, cleaning, analyzing, and visualizing agricultural data related to rural development schemes, specifically focused on Gujarat State's food grain crop estimates for the year 2023-24.

The key objectives include:
- Data collection from government portals
- Data cleaning and processing using Python (Pandas & NumPy)
- Visualization of crop-wise yield and production trends using Matplotlib and Seaborn
- Mini AI implementation for summarizing government scheme descriptions using Hugging Face transformers
- Prototype deployment using Streamlit for internal testing
- Documentation of insights and methodology

---

## Project Structure

rural-development-project/
│
├── data/ # Raw and processed datasets
├── notebooks/ # Jupyter notebooks for exploration and analysis
├── src/ # Source code: data processing, visualization, summarization, app
│ ├── data_processing.py
│ ├── visualization.py
│ ├── summarization.py
│ └── app.py # Streamlit prototype app
├── reports/ # Final project reports and documentation
├── requirements.txt # Python dependencies
└── README.md # Project overview and instructions



---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/rural-development-project.git
cd rural-development-project

python -m venv venv
source venv/bin/activate    # Linux/Mac
venv\Scripts\activate       # Windows

Usage
Data Cleaning & Analysis

Open and run the Jupyter notebook notebooks/data_cleaning_analysis.ipynb for step-by-step data processing, visualization, and analysis.

Summarization

Use the src/summarization.py module to run text summarization on government scheme descriptions with Hugging Face models.

Streamlit App

Run the Streamlit prototype interface for interactive data upload and text summarization:
streamlit run src/app.py

Key Insights

Crop-wise yield and productivity trends for Gujarat State's major food grains.

Identification of crops with highest production efficiency.

Summarization of complex government scheme texts for easier understanding.

Data Sources

Official agricultural and rural development datasets from data.gov.in
 and relevant Gujarat agricultural department portals.

Contact

For questions or collaboration, please contact:

Your Name
Email: your.email@example.com

License

This project is licensed under the MIT License.


---

Would you like me to help draft any specific sections like a sample analysis summary or instructions for contribution?