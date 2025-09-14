import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

st.set_page_config(page_title="EV Sales Clustering", layout="wide")

st.title("🔋 EV Sales Clustering App")
st.write("Upload your EV sales dataset and visualize clusters using K-Means.")

# --------------------------
# File Upload
# --------------------------
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Dataset Preview")
    st.dataframe(df.head(7))

    # --------------------------
    # Select Categorical Columns
    # --------------------------
    st.sidebar.header("⚙ Column Selection")
    categorical_cols = st.sidebar.multiselect(
        "Select categorical columns to encode",
        options=df.columns.tolist(),
        default=['Vehicle_Category', 'State', 'Vehicle_Class', 'Vehicle_Type', 'Month_Name']
        if all(col in df.columns for col in ['Vehicle_Category','State','Vehicle_Class','Vehicle_Type','Month_Name'])
        else []
    )

    numeric_cols = st.sidebar.multiselect(
        "Select numeric columns",
        options=df.columns.tolist(),
        default=['EV_Sales_Quantity'] if 'EV_Sales_Quantity' in df.columns else []
    )

    if categorical_cols or numeric_cols:
        df_encoded = df.copy()
        label_encoders = {}

        # Encode categorical columns
        for col in categorical_cols:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col])
            label_encoders[col] = le

        # Select features
        features = categorical_cols + numeric_cols
        X = df_encoded[features]

        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # --------------------------
        # Elbow Method + Silhouette Score
        # --------------------------
        inertia = []
        silhouette_scores = []
        k_values = range(2, 11)

        for k in k_values:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(X_scaled)
            inertia.append(kmeans.inertia_)
            score = silhouette_score(X_scaled, kmeans.labels_)
            silhouette_scores.append(score)

        st.subheader("📊 Optimal Cluster Selection")
        fig, ax = plt.subplots(1, 2, figsize=(14, 5))

        ax[0].plot(k_values, inertia, marker='o')
        ax[0].set_title("Elbow Method")
        ax[0].set_xlabel("Number of Clusters (k)")
        ax[0].set_ylabel("Inertia")

        ax[1].plot(k_values, silhouette_scores, marker='o')
        ax[1].set_title("Silhouette Scores")
        ax[1].set_xlabel("Number of Clusters (k)")
        ax[1].set_ylabel("Score")

        st.pyplot(fig)

        # --------------------------
        # User selects k
        # --------------------------
        chosen_k = st.sidebar.slider("Select number of clusters (k)", min_value=2, max_value=10, value=3)
        kmeans = KMeans(n_clusters=chosen_k, random_state=42)
        df['Cluster'] = kmeans.fit_predict(X_scaled)

        st.subheader(f"🔎 Clustering Results (k={chosen_k})")
        st.dataframe(df.head())

        # --------------------------
        # Visualization
        # --------------------------
        if len(numeric_cols) > 0:
            st.subheader("📈 Cluster Visualization")
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.scatterplot(x=df[numeric_cols[0]], y=df[categorical_cols[0]] if categorical_cols else df[numeric_cols[0]],
                            hue=df['Cluster'], palette="viridis", ax=ax)
            plt.title("K-Means Clustering")
            st.pyplot(fig)
    else:
        st.warning("Please select at least one categorical or numeric column from the sidebar.")
else:
    st.info("👆 Upload a CSV file to get started.")
