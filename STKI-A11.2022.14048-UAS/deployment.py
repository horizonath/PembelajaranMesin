import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon="📊",
    layout="wide",
)

# Header aplikasi
st.title("📊 Customer Segmentation Dashboard")
st.markdown("Dashboard ini digunakan untdf menganalisis data pelanggan menggunakan metode RFM (Recency, Frequency, Monetary).")


# Membaca dataset
data_file = "https://drive.google.com/drive/folders/1_ap0Gt6iTyzrXi1lPh7XcZG-un5hHNLn?usp=drive_link"
df = pd.read_csv(data_file, on_bad_lines="skip", encoding="unicode_escape")
st.write("### Dataset yang Diunggah")
st.dataframe(df.head())

# Deskripsi data
st.subheader("📋 Deskripsi Data")
st.write(df.describe())

# Missing Values
st.subheader("🚨 Missing Values")
st.write(df.isnull().sum())


# Data Cleaning
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
df = df[df["CustomerID"].notnull()]
df["Revenue"] = df["Quantity"] * df["UnitPrice"]


# Visualisasi distribusi Quantity
st.subheader("📊 Distribusi Quantity")
fig, ax = plt.subplots()
sns.histplot(df["Quantity"], kde=True, bins=50, ax=ax)
ax.set_title("Distribusi Quantity")
st.pyplot(fig)  


st.subheader("📊 Distribusi Unit Price")
fig, ax = plt.subplots()
sns.histplot(df["UnitPrice"], kde=False, bins=50, ax=ax)
ax.set_title("Distribusi Unit Price")
st.pyplot(fig)

st.subheader("🌍 5 Negara dengan Transaksi Terbesar")
fig, ax = plt.subplots()
top_countries = df["Country"].value_counts().head(5)
sns.barplot(x=top_countries.values, y=top_countries.index, ax=ax)
ax.set_title("5 Negara dengan Transaksi Terbesar")
st.pyplot(fig)

# Filter berdasarkan negara
if "Country" in df.columns:
    st.subheader("🌍 Pilih Negara untuk Analisis")
    countries = df["Country"].dropna().unique()
    selected_country = st.selectbox("Pilih Negara:", countries)
    country = df[df["Country"] == selected_country]
    st.write(f"### Data untuk Negara: {selected_country}")
    st.dataframe(country.head())



st.subheader("📈 Analisis Recency\n")

#Recency
df = pd.DataFrame(df['CustomerID'].unique())
df.columns = ['CustomerID']

latest_purchase = country.groupby('CustomerID').InvoiceDate.max().reset_index()
latest_purchase.columns = ['CustomerID','LatestPurchaseDate']

latest_purchase['Recency'] = (latest_purchase['LatestPurchaseDate'].max() - latest_purchase['LatestPurchaseDate']).dt.days

df = pd.merge(df, latest_purchase[['CustomerID','Recency']], on='CustomerID')


# Distribusi Recency

st.markdown("Plot Distribusi Recency\n\n")
fig, ax = plt.subplots()
sns.histplot(df["Recency"], kde=False, bins=20, ax=ax)
ax.set_title("Distribusi Recency")
st.pyplot(fig)


#Menentukan jumlah cluster
score = []
for k in range(1, 15):
    kmeans = KMeans(n_clusters=k,)
    member = kmeans.fit_predict(np.array(df['Recency']).reshape(-1, 1))
    score.append(kmeans.inertia_)


st.markdown("Menentukan jumlah cluster\n\n")
fig, ax = plt.subplots()
ax.plot(range(1, 15), score, marker="o")
ax.set_title("Recency Elbow Method")
ax.set_xlabel("Jumlah Cluster")
ax.set_ylabel("Inertia")
st.pyplot(fig)

#hitung receny score
kmeans = KMeans(n_clusters=4)
kmeans.fit(df[['Recency']])
df['RecencyCluster'] = kmeans.predict(df[['Recency']])


    #mengurutkan cluster
def order_cluster(cluster_field_name, target_field_name,df,ascending):
    new_cluster_field_name = 'new_' + cluster_field_name
    df_new = df.groupby(cluster_field_name)[target_field_name].mean().reset_index()
    df_new = df_new.sort_values(by=target_field_name,ascending=ascending).reset_index(drop=True)
    df_new['index'] = df_new.index
    df_final = pd.merge(df,df_new[[cluster_field_name,'index']], on=cluster_field_name)
    df_final = df_final.drop([cluster_field_name],axis=1)
    df_final = df_final.rename(columns={"index":cluster_field_name})
    return df_final


df = order_cluster('RecencyCluster', 'Recency',df,False)
st.markdown("Jumlah Cluster\n\n")
st.dataframe(df.head())






st.subheader("📈 Analisis Frequency")


#Frequency
frequency = country.groupby('CustomerID').InvoiceDate.count().reset_index()
frequency.columns = ['CustomerID','Frequency']


df = pd.merge(df, frequency, on='CustomerID')


st.markdown("Plot Distribusi Frequency\n\n")
fig, ax = plt.subplots()
sns.histplot(df["Frequency"], kde=False, bins=50, ax=ax)
ax.set_title("Distribusi Frequency")
st.pyplot(fig)


    #menetukan jumlah cluster
score = []
for k in range(1, 15):
    kmeans = KMeans(n_clusters=k)
    member = kmeans.fit_predict(np.array(df['Frequency']).reshape(-1, 1))
    score.append(kmeans.inertia_)

st.markdown("Menentukan Jumlah Cluster\n\n")
fig, ax = plt.subplots()
ax.plot(range(1, 15), score, marker="o")
ax.set_title("Recency Elbow Method")
plt.figure(figsize=(10, 5))
plt.plot(range(1, 15), score)
plt.ylabel("Inertia")
plt.xlabel("n_clusters")
st.pyplot(fig)


    #hitung frequency score
kmeans = KMeans(n_clusters=4)
kmeans.fit(df[['Frequency']])
df['FrequencyCluster'] = kmeans.predict(df[['Frequency']])


df = order_cluster('FrequencyCluster', 'Frequency',df,True)
st.markdown("Jumlah Cluster\n\n")
st.dataframe(df.head())











st.subheader("📈 Analisis Monetary")
#Monetary value (Revenue)
country['Revenue'] = country['UnitPrice'] * country['Quantity']
revenue = country.groupby('CustomerID').Revenue.sum().reset_index()

df = pd.merge(df, revenue, on='CustomerID')

st.markdown("Plot Distribusi Monetary\n\n")
fig, ax = plt.subplots()
sns.histplot(df["Revenue"], kde=False, bins=20, ax=ax)
ax.set_title("Distribusi Revenue")
st.pyplot(fig)

    #mentekuan jumlah cluster
score = []
for k in range(1, 15):
    kmeans = KMeans(n_clusters=k)
    member = kmeans.fit_predict(np.array(df['Revenue']).reshape(-1, 1))
    score.append(kmeans.inertia_)
    

st.markdown("Menentukan Jumlah Cluster\n\n")
fig, ax = plt.subplots()
ax.set_title("Revenue Elbow Chart")
ax.plot(range(1, 15), score, marker="o")
plt.figure(figsize=(10, 5))
plt.plot(range(1, 15), score)
plt.ylabel("Inertia")
plt.xlabel("n_clusters")
st.pyplot(fig)


    #hitung monetary score
kmeans = KMeans(n_clusters=4)
kmeans.fit(df[['Revenue']])
df['RevenueCluster'] = kmeans.predict(df[['Revenue']])
df.head()


df = order_cluster('RevenueCluster', 'Revenue',df,True)
st.markdown("Jumlah Cluster\n\n")
st.dataframe(df.head())








st.subheader("📈 Analisis RFM")
#Hitung RFM secara keseluruhan

# RFM Analysis
df['RFM_score'] = df['RecencyCluster'] + df['FrequencyCluster'] + df['RevenueCluster']


df['label'] = 'Bronze' 
df.loc[df['RFM_score'] > 1, 'label'] = 'Silver' 
df.loc[df['RFM_score'] > 2, 'label'] = 'Gold'
df.loc[df['RFM_score'] > 3, 'label'] = 'Platinum'
df.loc[df['RFM_score'] > 5, 'label'] = 'Diamond'


fig, ax = plt.subplots()
barplot = dict(df['label'].value_counts())
bar_names = list(barplot.keys())
bar_values = list(barplot.values())
plt.bar(bar_names,bar_values)
st.pyplot(fig)

print(pd.DataFrame(barplot, index=[' ']))




X_rfm = df[['Recency', 'Frequency', 'Revenue']]

kmeans_rfm = KMeans(n_clusters=4, random_state=42)
kmeans_rfm.fit(X_rfm)

labels_rfm = kmeans_rfm.labels_

sil_score_rfm = silhouette_score(X_rfm, labels_rfm)
st.caption(f"Silhouette Score for RFM clustering: {sil_score_rfm:.4f}\n")



st.markdown(
""" **Strategi Pemasaran:**\n
**Platinum dan Diamond:** Fokus pada retensi pelanggan dengan promosi eksklusif.\n
**Silver dan Gold:** Tingkatkan frekuensi pembelian dengan penawaran khusus.\n
**Bronze:** Libatkan pelanggan melalui kampanye promosi untuk meningkatkan pendapatan.\n"""
)
