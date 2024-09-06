# Credit Card Customer Segmentation Project

## Project Overview
This project focuses on segmenting credit card customers based on their spending patterns and behaviors. The goal is to help a financial institution develop targeted marketing strategies and better understand customer needs using unsupervised machine learning techniques. 

We performed clustering using **K-Means** after reducing the dimensionality of the dataset with **Principal Component Analysis (PCA)**. The project involved multiple steps, including data preprocessing, feature engineering, model building, and validation.

## Table of Contents
1. [Data Overview](#data-overview)
2. [Data Preprocessing](#data-preprocessing)
3. [Dimensionality Reduction (PCA)](#dimensionality-reduction-pca)
4. [Clustering with K-Means](#clustering-with-k-means)
5. [Model Validation](#model-validation)
6. [Results and Interpretation](#results-and-interpretation)
7. [Technical Skills](#technical-skills)

---

## Data Overview

The dataset contains behavioral data on 9,000 active credit card holders over six months. The key columns include:

- **CUST_ID**: Unique identifier for each customer.
- **BALANCE**: Current balance on the credit card.
- **PURCHASES**: Total amount spent using the card.
- **ONEOFF_PURCHASES**: Largest single purchase made.
- **INSTALLMENTS_PURCHASES**: Total amount spent on installment-based purchases.
- **CASH_ADVANCE**: Total cash withdrawn using the card.
- **PURCHASES_FREQUENCY**: Frequency of purchases.
- **CREDIT_LIMIT**: Maximum limit on the credit card.
- **PAYMENTS**: Total amount paid by the customer.
- **TENURE**: Duration of the customer’s relationship with the credit card company.

---

## Data Preprocessing

To ensure the quality of the input data, the following preprocessing steps were implemented:

### 1. Handling Missing Values
We identified missing values in the **MINIMUM_PAYMENTS** and **CREDIT_LIMIT** columns. The missing data was imputed using:
- **KNN Imputer**: This method imputed missing values based on the closest neighbors, ensuring the imputed values were aligned with the overall customer profiles.

### 2. Feature Scaling
The features had different units and scales, which could negatively impact the clustering model. To standardize the features:
- **StandardScaler**: We applied standardization to ensure each feature had a mean of 0 and a standard deviation of 1, making them comparable.

### 3. Handling Skewed Data
Certain features like **CASH_ADVANCE** and **BALANCE** exhibited right skewness. We applied logarithmic transformation to reduce skewness and improve model performance.

### 4. Dimensionality Reduction (PCA)
With 18 features, there was a need to reduce dimensionality for better visualization and model performance. We applied **Principal Component Analysis (PCA)**:
- **Explained Variance**: We selected the number of components based on the explained variance ratio. The first 3 components explained around 51.33% of the variance.
- **Scree Plot**: The scree plot helped us decide on the optimal number of components.

### 5. Creating New Features
To enhance the model's ability to segment customers, we created derived features:
- **Balance to Credit Ratio**: Indicates how much of the credit limit is being utilized.
- **Purchase to Payment Ratio**: Provides insights into customer repayment behavior.

---

## Clustering with K-Means

To segment customers, we used **K-Means Clustering**, an unsupervised machine learning algorithm.

### 1. Determining Optimal Clusters
- **Elbow Method**: We used the elbow method to determine the optimal number of clusters by plotting the sum of squared distances (inertia) against the number of clusters. The elbow point was found at 7 clusters.
  
- **Silhouette Score**: The average silhouette score was used to evaluate how well-separated the clusters were. A silhouette score of 0.42 indicated reasonable separation between clusters, signifying that the clustering was meaningful.

### 2. Model Training
We trained the K-Means model with **k=7** clusters based on the elbow method. The clusters were assigned based on customer spending patterns across various features.

### DBSCAN (Density-Based Clustering)

In addition to **K-Means**, we experimented with **DBSCAN** (Density-Based Spatial Clustering of Applications with Noise), a powerful clustering algorithm suited for identifying clusters of varying shapes and sizes, as well as outliers. DBSCAN groups points based on the density of their neighborhood. It requires two key parameters: **epsilon** (the maximum distance between two points for them to be considered neighbors) and **min_samples** (the minimum number of points to form a cluster).

Through several trials of tuning epsilon and min_samples, we found that DBSCAN was useful for detecting dense regions and outliers, but due to the evenly spread customer data, it resulted in many points being labeled as noise.

---

### Agglomerative Clustering

We also explored **Agglomerative Clustering**, a hierarchical clustering technique that builds clusters in a bottom-up manner. Each point starts as its own cluster, and pairs of clusters are merged based on proximity, using linkage criteria such as **ward linkage**, **average linkage**, or **complete linkage**.

The strength of Agglomerative Clustering lies in the ability to visualize clustering results using **dendrograms**, which allowed us to observe how data points merged into clusters at different distances. Though it is more computationally expensive than K-Means, Agglomerative Clustering provided insight into the hierarchical structure of the data and relationships between the clusters, offering a unique perspective for segmentation.


---

## Model Validation

The model was validated using the following techniques:

- **Silhouette Score**: Provided an average score of 0.42, showing reasonable separation between clusters.
- **Cluster Visualization**: We used 2D scatter plots and principal components to visualize the clusters and their separation.

---

## Results and Interpretation

The K-Means clustering algorithm segmented customers into 7 clusters, each representing distinct spending behaviors.

### 1. Cluster 1: **High Installment & Frequency Users**
- These customers prefer frequent installment payments and rarely make large one-time purchases.
- **Strategy**: Offer personalized installment plans and loyalty programs to retain them.

### 2. Cluster 2: **One-Time Payment Users**
- Customers who favor large one-time purchases and seldom use installment options.
- **Strategy**: Incentivize upfront payments through limited-time offers and discounts.

### 3. Cluster 3: **Moderate Installment Users**
- A balanced approach between one-time and installment payments.
- **Strategy**: Educate customers on the benefits of installment options.

### 4. Cluster 4: **Cash-Dependent Users**
- Customers who frequently withdraw cash advances and prefer installment payments.
- **Strategy**: Introduce cash-back incentives for installment purchases to reduce cash dependency.

### 5. Cluster 5: **Mixed Payment Users**
- A mixture of one-time and installment payments with moderate cash advances.
- **Strategy**: Offer flexible payment plans and product bundles.

### 6. Cluster 6: **High Cash-Advance Users**
- Heavy reliance on cash withdrawals with occasional installment purchases.
- **Strategy**: Promote financial literacy and responsible spending habits.

### 7. Cluster 7: **High Frequency Cash-Advance Users**
- Frequent installment users with a moderate preference for cash withdrawals.
- **Strategy**: Implement exclusive membership programs with targeted promotions.

---

## Technical Skills

- **Python**: For implementing the entire workflow, including data preprocessing and modeling.
- **Pandas & NumPy**: For data manipulation, feature engineering, and numerical operations.
- **Scikit-Learn**: For applying PCA, K-Means, and evaluation metrics like the silhouette score.
- **Matplotlib & Seaborn**: For visualizing the scree plot, elbow plot, and cluster separations.
- **KNN Imputer**: For imputing missing values in critical features.
- **StandardScaler**: For feature standardization.
- **K-Means Clustering**: For customer segmentation.
- **Principal Component Analysis (PCA)**: For dimensionality reduction.

---

## Conclusion

This project successfully segmented credit card customers based on their spending patterns, providing actionable insights for targeted marketing. The combination of **PCA** and **K-Means Clustering** helped in identifying distinct customer behaviors, enabling personalized marketing strategies. This segmentation can help financial institutions better serve their customers by aligning their products with customer needs and preferences.

