import streamlit as st
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from numpy import array
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import altair as alt
from sklearn.utils.validation import joblib
from sklearn.preprocessing import StandardScaler
from PIL import Image

# display
st.set_page_config(page_title="Liver Prediction", page_icon='icon.jpg')

primaryColor="#6eb52f"
backgroundColor="yellow"
secondaryBackgroundColor="#e0e0ef"
textColor="#262730"
font="sans serif"

st.title("Sistem Prediksi Penyakit Liver")
st.write("Aprilia Nazwa Ambarak E13.2022.00202")
st.write("Sistem Cerdas")
st.write("========================================================================================")
data, implementation = st.tabs(["Data", "Implementation"])

df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/00225/Indian%20Liver%20Patient%20Dataset%20(ILPD).csv")
# Memberikan nama fitur pada tiap kolom dataset
df.columns = ['Age', 'Gender', 'Total_Bilirubin', 'Direct_Bilirubin', 'Alkaline_Phosphotase',
                'Alamine_Aminotransferase', 'Aspartate_Aminotransferase', 'Total_Protiens',
                'Albumin', 'Albumin_and_Globulin_Ratio', 'Liver_disease']


with data:
    st.write("""# Tentang Dataset dan Aplikasi""")

    st.write("Dataset yang digunakan adalah Indian Patient Liver dataset yang diambil dari https://archive.ics.uci.edu/ml/datasets/ILPD+(Indian+Liver+Patient+Dataset)")
    st.write("Total datanya adalah 582 dengan data training 90% (523) dan data testing 10% (59)")

with implementation:
    st.write("# Implementation")
  
    df = df.fillna(df.mean(numeric_only=True))

    # Buat dictionary untuk melakukan mapping variabel Liver_disease
    liver = {'Liver_disease': {1: 1, 2: 0}}

    df = df.replace(liver)

   # Buat dictionary untuk melakukan mapping variabel kategorikal menjadi variabel numerikal
    gender = {'Gender': {"Male": 1, "Female": 0}}

    df = df.replace(gender)
    
    X = df.drop(columns="Liver_disease")
    y = df.Liver_disease
   
   

    labels = pd.get_dummies(df.Liver_disease).columns.values.tolist()
 
    scaler = MinMaxScaler()
    scaler.fit(X)
    X = scaler.transform(X)
 
    
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.1,random_state=42)
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    # SVM
    kernel = 'linear'
    svm_model = SVC(kernel='linear')
    svm_model.fit(X_train, y_train)
    y_pred=svm_model.predict(X_test)

    skor_akurasi = round(100 * accuracy_score(y_test,y_pred))

    # RF

    rf_model = RandomForestClassifier()

    # Melatih model dengan data latih
    rf_model.fit(X_train, y_train)
    # prediction
    rf_model.score(X_test, y_test)
    y_pred = rf_model.predict(X_test)
    #Accuracy
    akurasiii = round(100 * accuracy_score(y_test,y_pred))

    Age = st.number_input('Masukkan Umur Pasien')

    # GENDER
    gender = st.radio("Gender",('Male', 'Female'))
    if gender == "Male":
        Gender = 1
    elif gender == "Female" :
        Gender = 0

    Total_Bilirubin = st.number_input('Masukkan Hasil Test Total_Bilirubin (Contoh : 10.9)')
    Direct_Bilirubin = st.number_input('Masukkan Hasil Test Direct_Bilirubin (Contoh : 5.5)')
    Alkaline_Phosphotase = st.number_input('Masukkan Hasil Test Alkaline_Phosphotase (Contoh : 699)')
    Alamine_Aminotransferase = st.number_input('Masukkan Hasil Test Alamine_Aminotransferase (Contoh : 64)')
    Aspartate_Aminotransferase = st.number_input('Masukkan Hasil Test Aspartate_Aminotransferase (Contoh : 100)')
    Total_Protiens = st.number_input('Masukkan Hasil Test Total_Protiens (Contoh : 7.5)')
    Albumin = st.number_input('Masukkan Hasil Test Albumin (Contoh : 3.2)')
    Albumin_and_Globulin_Ratio = st.number_input('Masukkan Hasil Albumin_and_Globulin_Ratio (Contoh : 0.74)')
   



    def submit():
        # input
        inputs = np.array([[
            Age, Gender, Total_Bilirubin, Direct_Bilirubin, Alkaline_Phosphotase,
                Alamine_Aminotransferase, Aspartate_Aminotransferase, Total_Protiens,
                Albumin, Albumin_and_Globulin_Ratio
            ]])
        le = joblib.load("le.save")

        if skor_akurasi > akurasiii:
            model = joblib.load("svm_model.joblib")

        elif akurasiii > skor_akurasi:
            model = joblib.load("rf_model.joblib")

        y_pred3 = model.predict(inputs)
        st.write(f"Berdasarkan data yang Anda masukkan, maka anda diprediksi cenderung : {le.inverse_transform(y_pred3)[0]}")
        st.write("0 = Tidak Terkena Liver")
        st.write("1 = Terkena Liver")
    all = st.button("Submit")
    if all :
        st.balloons()
        submit()
