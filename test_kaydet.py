import pandas as pd
import joblib
from tensorflow.keras.models import load_model
import os

MODEL_DIR = "model_ciktilari"
model = load_model(f"{MODEL_DIR}/musteri_kaybi_modeli.h5")
scaler = joblib.load(f"{MODEL_DIR}/olceklendirici.pkl")
threshold = joblib.load(f"{MODEL_DIR}/optimal_esik.pkl")
feature_names = joblib.load(f"{MODEL_DIR}/ozellik_isimleri.pkl")

df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

def feature_engineering(df):
    df = df.copy()
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)

    for col in ['OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']:
        df[col] = df[col].replace({'No internet service': 'No'})

    df['SözleşmeSüreci'] = df['tenure'].apply(lambda x: 'Kısa' if x < 12 else ('Orta' if x < 24 else 'Uzun'))
    df['tenure_kat'] = df['tenure'].apply(lambda x: 'Kısa' if x < 12 else ('Orta' if x < 24 else 'Uzun'))
    df['monthly_kat'] = df['MonthlyCharges'].apply(lambda x: 'Düşük' if x < 35 else ('Orta' if x < 70 else 'Yüksek'))

    hizmet_kolonlari = ['PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']

    df['HizmetSayısı'] = df[hizmet_kolonlari].apply(lambda row: sum(x == 'Yes' for x in row), axis=1)
    df['HizmetYoğunluğu'] = df['HizmetSayısı'] / 9
    df['HizmetBaşıMaliyet'] = df['MonthlyCharges'] / (df['HizmetSayısı'] + 1)
    df['OrtalamaSözlesmeÜcreti'] = df['MonthlyCharges'] / (df['tenure'] + 1)
    df['HizmetDeğeri'] = df['TotalCharges'] / (df['MonthlyCharges'] + 1)

    df = pd.get_dummies(df, columns=[
        'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
        'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
        'PaperlessBilling', 'PaymentMethod', 'SözleşmeSüreci', 'tenure_kat', 'monthly_kat'
    ])

    for col in feature_names:
        if col not in df.columns:
            df[col] = 0

    return df[feature_names]

# Sonuçları listele
sonuclar = []

for i, row in df.head(100).iterrows():  # İlk 50 satırı al
    processed = feature_engineering(pd.DataFrame([row]))
    scaled = scaler.transform(processed)
    prob = model.predict(scaled)[0][0]
    pred = int(prob >= threshold)
    sonuc = {
        "Müşteri ID": row["customerID"],
        "Gerçek Durum": row["Churn"],
        "Tahmin": "Ayrılacak" if pred == 1 else "Kalacak",
        "Olasılık (%)": round(prob * 100, 2)
    }
    print(sonuc)
    sonuclar.append(sonuc)

# Excel'e yaz
output_df = pd.DataFrame(sonuclar)
output_df.to_excel("tahmin_sonuclari.xlsx", index=False)
print("\n✅ Tahmin sonuçları 'tahmin_sonuclari.xlsx' dosyasına kaydedildi.")
