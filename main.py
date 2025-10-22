
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def ensure_dirs():
    os.makedirs("output_graphs", exist_ok=True)

def load_txn(path="transactions.csv"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Make sure the CSV is in this folder.")
    return pd.read_csv(path)

def build_rfm(df):
    # Ensure Date is datetime
    df['Date'] = pd.to_datetime(df['Date'])
    ref_date = pd.Timestamp('2024-12-31')
    rfm = df.groupby('CustomerID').agg(
        Frequency=('Total','count'),
        Monetary=('Total','sum'),
        LastPurchase=('Date','max')
    ).reset_index()
    rfm['Recency'] = (ref_date - rfm['LastPurchase']).dt.days
    rfm = rfm.drop(columns=['LastPurchase'])
    return rfm

def cluster_rfm(rfm, k=4):
    features = rfm[['Frequency','Monetary','Recency']].copy()
    model = KMeans(n_clusters=k, random_state=10, n_init=10)
    rfm['Cluster'] = model.fit_predict(features)
    return rfm, model

def save_plot(rfm):
    plt.figure()
    plt.scatter(rfm['Frequency'], rfm['Monetary'])
    plt.xlabel('Frequency'); plt.ylabel('Monetary'); plt.title('Customer Segmentation – Monetary vs Frequency')
    plt.savefig('output_graphs/monetary_vs_frequency.png', bbox_inches='tight')
    plt.close()

def main():
    ensure_dirs()
    df = load_txn()
    rfm = build_rfm(df)
    rfm, model = cluster_rfm(rfm, k=4)
    save_plot(rfm)
    rfm.to_csv('rfm_clusters.csv', index=False)
    print("Saved: output_graphs/monetary_vs_frequency.png and rfm_clusters.csv")
    print(rfm.head())

if __name__ == "__main__":
    main()
