from flask import Flask, render_template, request, redirect, url_for, flash
import pandas as pd
import numpy as np
import joblib
import os
from tensorflow.keras.models import load_model
import base64
from io import BytesIO
import matplotlib
matplotlib.use('Agg')  # GUI olmayan ortamlarda çizim için
import matplotlib.pyplot as plt

app = Flask(__name__)
app.secret_key = "telco_musteri_siniflandirma_projesi"

MODEL_DIR = 'model_ciktilari'
model = None
scaler = None
optimal_threshold = None
feature_names = None

# Model bileşenlerini yükle

def load_model_components():
    global model, scaler, optimal_threshold, feature_names
    try:
        model = load_model(f'{MODEL_DIR}/musteri_kaybi_modeli.h5')
        scaler = joblib.load(f'{MODEL_DIR}/olceklendirici.pkl')
        optimal_threshold = joblib.load(f'{MODEL_DIR}/optimal_esik.pkl')
        feature_names = joblib.load(f'{MODEL_DIR}/ozellik_isimleri.pkl')
        print("✅ Model bileşenleri başarıyla yüklendi.")
    except Exception as e:
        print(f"❌ Model yükleme hatası: {e}")

load_model_components()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/musteri-formu')
def musteri_formu():
    return render_template('musteri_formu.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/rapor')
def rapor():
    try:
        with open(f"{MODEL_DIR}/siniflandirma_raporu.txt", "r") as file:
            rapor_icerik = file.read()
        return render_template("report.html", rapor=rapor_icerik)
    except Exception as e:
        flash(f"Rapor yüklenemedi: {e}", "danger")
        return redirect(url_for('index'))

@app.route('/siniflandirma_sonuc', methods=['GET', 'POST'])
def siniflandirma_sonuc():
    if request.method == 'GET':
        flash("Lütfen önce müşteri formunu doldurun.", "warning")
        return redirect(url_for('musteri_formu'))

    try:
        form_data = request.form.to_dict()
        form_data['SeniorCitizen'] = int(form_data.get('SeniorCitizen', 0))
        form_data['tenure'] = int(form_data.get('tenure', 0))
        form_data['MonthlyCharges'] = float(form_data.get('MonthlyCharges', 0))
        form_data['TotalCharges'] = form_data['tenure'] * form_data['MonthlyCharges']

        df = pd.DataFrame([form_data])
        df = feature_engineering(df)
        prediction, probability = make_prediction(df)

        risk_level = get_risk_level(probability[0][0])
        risk_chart = create_risk_chart(probability[0][0])

        return render_template('siniflandirma_sonuc.html',
                               prediction=prediction[0][0],
                               probability=probability[0][0],
                               risk_level=risk_level,
                               risk_chart=risk_chart,
                               customer_data=form_data)
    except Exception as e:
        flash(f"Hata oluştu: {e}", "danger")
        return redirect(url_for('musteri_formu'))

# Özellik mühendisliği

def feature_engineering(df):
    df_processed = df.copy()

    # Güvenli dönüşümler (varsa uygula)
    for col in ['OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']:
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].replace({'No internet service': 'No'})

    # Ek özellikler
    df_processed['OrtalamaSözlesmeÜcreti'] = df_processed['MonthlyCharges'] / (df_processed['tenure'] + 1)
    df_processed['HizmetDeğeri'] = df_processed['TotalCharges'] / (df_processed['MonthlyCharges'] + 1)

    # One-hot encoding
    df_processed = pd.get_dummies(df_processed)

    # Eksik olan sütunları tamamla
    for col in feature_names:
        if col not in df_processed.columns:
            df_processed[col] = 0

    # Sıra bozulmasın
    return df_processed[feature_names]

    df = df.copy()
    df['OrtalamaSözlesmeÜreti'] = df['MonthlyCharges'] / (df['tenure'] + 1)
    df['HizmetDeğeri'] = df['TotalCharges'] / (df['MonthlyCharges'] + 1)
    df['SözleşmeSüreci'] = df['tenure'].apply(lambda x: 'Kısa' if x < 12 else ('Orta' if x < 24 else 'Uzun'))
    hizmet_kolonlari = ['PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    df['HizmetSayısı'] = df[hizmet_kolonlari].apply(lambda row: sum(x == 'Yes' for x in row), axis=1)
    df['HizmetYoğunluğu'] = df['HizmetSayısı'] / 9
    df['HizmetBaşıMaliyet'] = df['MonthlyCharges'] / (df['HizmetSayısı'] + 1)
    df['tenure_kat'] = df['tenure'].apply(lambda x: 'Kısa' if x < 12 else ('Orta' if x < 24 else 'Uzun'))
    df['monthly_kat'] = df['MonthlyCharges'].apply(lambda x: 'Düşük' if x < 35 else ('Orta' if x < 70 else 'Yüksek'))

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

# Tahmin fonksiyonu

def make_prediction(df):
    X_scaled = scaler.transform(df)
    prob = model.predict(X_scaled)
    pred = (prob >= optimal_threshold).astype(int)
    return pred, prob

# Risk seviyesi

def get_risk_level(prob):
    if prob < 0.3:
        return "Düşük"
    elif prob < 0.6:
        return "Orta"
    return "Yüksek"

# Risk grafiği

def create_risk_chart(prob):
    fig, ax = plt.subplots(figsize=(4, 2))
    ax.barh(['Risk Skoru'], [prob], color='red' if prob > 0.6 else 'orange' if prob > 0.3 else 'green')
    plt.xlim(0, 1)
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    return img

if __name__ == '__main__':
    os.makedirs('static/images', exist_ok=True)
    app.run(debug=True)