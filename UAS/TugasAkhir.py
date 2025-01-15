import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from lifetimes.utils import summary_data_from_transaction_data

# Header aplikasi
st.title("Customer Segmentation Dashboard")
st.markdown("""
Dashboard ini digunakan untuk menganalisis data pelanggan menggunakan metode RFM (Recency, Frequency, Monetary).
""")

# Input dataset
data_file = st.file_uploader("Unggah dataset CSV", type="csv")
if data_file is not None:
    df = pd.read_csv(data_file, on_bad_lines='skip', encoding='unicode_escape')
    st.write("### Dataset yang diunggah:")
    st.dataframe(df.head())

    # Data preprocessing
    st.subheader("Deskripsi Data")
    st.write(df.describe())

    # Visualisasi Missing Value
    st.subheader("Missing Values")
    st.write(df.isnull().sum())

    # Data Cleaning
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df = df[~df['CustomerID'].isna()]
    df = df[df['Quantity'] > 0]
    df = df[df['UnitPrice'] > 0]
    df['Revenue'] = df['Quantity'] * df['UnitPrice']

    # Pilih negara untuk analisis
    st.subheader("Pilih Negara untuk Analisis")
    unique_countries = df['Country'].unique()
    selected_country = st.selectbox("Pilih negara:", unique_countries)

    # Filter dataset berdasarkan negara yang dipilih
    filtered_df = df[df['Country'] == selected_country]
    st.write(f"### Data untuk negara: {selected_country}")
    st.dataframe(filtered_df.head())

    # RFM Analysis
    rfm = summary_data_from_transaction_data(filtered_df, 'CustomerID', 'InvoiceDate', monetary_value_col='Revenue').reset_index()
 
    #membuang data customer yang tidak melakukan pembelian ulang.
    rfm = rfm[rfm['frequency']>0]
    rfm.head()

    #Membatasi customer yang memiliki moetery value di atas 2000
    rfm = rfm[rfm['monetary_value']<2000]
    st.dataframe(rfm.head())

    #membagi Score menjadi empat bagian
    quartiles = rfm.quantile(q=[0.25, 0.5, 0.75])
    quartiles

    # Scoring Functions
    def recency_score(data):
        if data <= 60:
            return 1
        elif data <= 128:
            return 2
        elif data <= 221:
            return 3
        else:
            return 4

    def frequency_score(data):
        if data <= 1:
            return 1
        elif data <= 2:
            return 2
        elif data <= 3:
            return 3
        else:
            return 4

    def monetary_value_score(data):
        if data <= 142.935:
            return 1
        elif data <= 292.555:
            return 2
        elif data <= 412.435:
            return 3
        else:
            return 4

    rfm['R'] = rfm['recency'].apply(recency_score)
    rfm['F'] = rfm['frequency'].apply(frequency_score)
    rfm['M'] = rfm['monetary_value'].apply(monetary_value_score)

    rfm['RFM_score'] = rfm[['R', 'F', 'M']].sum(axis=1)

    rfm['label'] = 'Bronze'
    rfm.loc[rfm['RFM_score'] > 4, 'label'] = 'Silver'
    rfm.loc[rfm['RFM_score'] > 6, 'label'] = 'Gold'
    rfm.loc[rfm['RFM_score'] > 8, 'label'] = 'Platinum'
    rfm.loc[rfm['RFM_score'] > 10, 'label'] = 'Diamond'

    # Display RFM Table
    st.subheader("Hasil RFM Analysis")
    st.write(rfm.head())

    # Visualisasi
    st.subheader("Distribusi Label")
    barplot = dict(rfm['label'].value_counts())
    bar_names = list(barplot.keys())
    bar_values = list(barplot.values())
    plt.bar(bar_names, bar_values)
    plt.title("Distribusi RFM Labels")
    plt.xlabel("Label")
    plt.ylabel("Jumlah Pelanggan")
    st.pyplot(plt)

else:
    st.write("Silakan unggah dataset untuk memulai analisis.")
