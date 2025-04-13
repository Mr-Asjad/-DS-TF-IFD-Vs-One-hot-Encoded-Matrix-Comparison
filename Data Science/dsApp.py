import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer
import matplotlib.pyplot as plt
import seaborn as sns

# Set up the Streamlit app
st.set_page_config(layout="wide")
st.title("Disease Classification: TF-IDF vs One-Hot Encoding")

# File uploader
uploaded_file = st.file_uploader("Upload a preprocessed CSV file", type="csv")

if uploaded_file:
    # Load CSV file
    df = pd.read_csv(uploaded_file)
    st.success("File uploaded successfully!")
    
    # Sidebar settings
    st.sidebar.subheader("🔧 Model Settings")
    selected_k = st.sidebar.selectbox("Select the value of k for KNN", [3, 5, 7])
    selected_metric = st.sidebar.selectbox("Select the distance metric for KNN", ['euclidean', 'manhattan', 'cosine'])

    # Prepare the data
    st.write("### Data Preview")
    st.dataframe(df.head())

    # Combine features
    df['combined_text'] = df['Risk Factors'].fillna('') + ' ' + df['Symptoms'].fillna('') + ' ' + df['Signs'].fillna('')

    # TF-IDF Vectorization
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(df['combined_text']).toarray()

    # One-Hot Encoding
    mlb = MultiLabelBinarizer()
    onehot_matrix = np.concatenate(
        [mlb.fit_transform(df[col].apply(lambda x: eval(x) if isinstance(x, str) else [])) for col in ['Risk Factors', 'Symptoms', 'Signs']],
        axis=1
    )

    # Target labels
    category_mapping = {
        "Acute Coronary Syndrome": "Cardiovascular",
        "Adrenal Insufficiency": "Endocrine",
        "Alzheimer": "Neurological",
        "Aortic Dissection": "Cardiovascular",
        "Asthma": "Respiratory",
        "Atrial Fibrillation": "Cardiovascular",
        "Cardiomyopathy": "Cardiovascular",
        "COPD": "Respiratory",
        "Diabetes": "Endocrine",
        "Epilepsy": "Neurological",
        "Gastritis": "Gastrointestinal",
        "Gastro-oesophageal Reflux Disease": "Gastrointestinal",
        "Heart Failure": "Cardiovascular",
        "Hyperlipidemia": "Cardiovascular",
        "Hypertension": "Cardiovascular",
        "Migraine": "Neurological",
        "Multiple Sclerosis": "Neurological",
        "Peptic Ulcer Disease": "Gastrointestinal",
        "Pituitary Disease": "Endocrine",
        "Pneumonia": "Respiratory",
        "Pulmonary Embolism": "Cardiovascular",
        "Stroke": "Neurological",
        "Thyroid Disease": "Endocrine",
        "Tuberculosis": "Infectious",
        "Upper Gastrointestinal Bleeding": "Gastrointestinal"
    }

    # Assign categories based on the mapping
    df['Category'] = df['Disease'].map(category_mapping)

    target = df['Category']

    # Define a function for model evaluation
    def evaluate_models(tfidf_matrix, onehot_matrix, target):
        scoring = {
            'accuracy': make_scorer(accuracy_score),
            'precision': make_scorer(precision_score, average='weighted', zero_division=0),
            'recall': make_scorer(recall_score, average='weighted', zero_division=0),
            'f1': make_scorer(f1_score, average='weighted', zero_division=0)
        }
        results = []
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

        for feature_name, feature_matrix in [('TF-IDF', tfidf_matrix), ('One-Hot', onehot_matrix)]:
            # KNN Model
            knn = KNeighborsClassifier(n_neighbors=selected_k, metric=selected_metric, weights='distance')
            knn_scores = cross_validate(knn, feature_matrix, target, cv=cv, scoring=scoring)
            results.append({
                'Model': 'KNN',
                'Feature': feature_name,
                'Accuracy': np.mean(knn_scores['test_accuracy']),
                'Precision': np.mean(knn_scores['test_precision']),
                'Recall': np.mean(knn_scores['test_recall']),
                'F1-Score': np.mean(knn_scores['test_f1'])
            })

            # Logistic Regression
            lr = LogisticRegression(max_iter=2000, class_weight='balanced')
            lr_scores = cross_validate(lr, feature_matrix, target, cv=cv, scoring=scoring)
            results.append({
                'Model': 'Logistic Regression',
                'Feature': feature_name,
                'Accuracy': np.mean(lr_scores['test_accuracy']),
                'Precision': np.mean(lr_scores['test_precision']),
                'Recall': np.mean(lr_scores['test_recall']),
                'F1-Score': np.mean(lr_scores['test_f1'])
            })

        return pd.DataFrame(results)

    # Evaluate models
    st.spinner("Evaluating models...")
    results_df = evaluate_models(tfidf_matrix, onehot_matrix, target)
    st.write("### Model Results")
    st.dataframe(results_df)

    # Visualize results
    st.write("### Model Performance by F1-Score")
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Feature', y='F1-Score', hue='Model', data=results_df)
    plt.title("Model Comparison (F1-Score)")
    st.pyplot(plt)
else:
    st.warning("Please upload a CSV file to proceed.")
