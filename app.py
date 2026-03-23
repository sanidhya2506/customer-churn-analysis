import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="Customer Segmentation App 📊",
    layout="wide"
)

st.title("Customer Segmentation App 📊")
st.markdown("Upload your customer data CSV to perform KMeans clustering.")

# Upload CSV
uploaded_file = st.file_uploader("Upload your customer CSV", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.subheader("Data Preview")
    st.dataframe(data.head())

    # Filter numeric columns only
    numeric_cols = data.select_dtypes(include='number').columns.tolist()
    if not numeric_cols:
        st.error("No numeric columns found for clustering. Please upload proper numeric data.")
    else:
        st.subheader("Select features for clustering")
        features = st.multiselect("Numeric features only", numeric_cols, default=numeric_cols[:2])

        if features:
            X = data[features]
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Number of clusters
            n_clusters = st.slider("Select number of clusters", 2, 10, 3)

            # Run KMeans
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            data['Cluster'] = kmeans.fit_predict(X_scaled)

            # Show results
            st.subheader("Clustered Data")
            st.dataframe(data.head())

            st.subheader("Cluster Counts")
            st.bar_chart(data['Cluster'].value_counts().sort_index())

            # 2D visualization if exactly 2 features
            if len(features) == 2:
                st.subheader("Cluster Plot")
                plt.figure(figsize=(8, 6))
                palette = sns.color_palette("bright", n_clusters)
                sns.scatterplot(
                    x=data[features[0]],
                    y=data[features[1]],
                    hue=data['Cluster'],
                    palette=palette,
                    s=100,
                    alpha=0.8
                )
                plt.xlabel(features[0])
                plt.ylabel(features[1])
                plt.title("Customer Segments")
                plt.legend(title="Cluster")
                st.pyplot(plt)
            else:
                st.info("Select exactly 2 features to visualize clusters.")