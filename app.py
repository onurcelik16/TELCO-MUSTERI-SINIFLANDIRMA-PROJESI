from flask import Flask, render_template, request, redirect, url_for, flash
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
from tensorflow.keras.models import load_model
import base64
from io import BytesIO
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import re

app = Flask(__name__)
app.secret_key = "telco_musteri_siniflandirma_projesi"

MODEL_DIR = 'model_ciktilari'
model = None
scaler = None
optimal_threshold = None
feature_names = None

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
    try:
        rapor_path = os.path.join(MODEL_DIR, "siniflandirma_raporu.txt")
        with open(rapor_path, "r") as file:
            lines = file.readlines()

        lines = [line.strip() for line in lines if line.strip()]

        for line in lines:
            if line.startswith("accuracy"):
                match = re.findall(r"[\d.]+", line)
                if match:
                    accuracy = float(match[0])
                    break
        else:
            raise ValueError("Accuracy değeri bulunamadı.")

        macro_line = next(line for line in lines if line.startswith("macro avg"))
        macro_values = re.findall(r"[\d.]+", macro_line)
        if len(macro_values) < 3:
            raise ValueError("macro avg satırında eksik veri var.")
        precision, recall, f1_score = map(float, macro_values[:3])

        metrics = {
            "accuracy": round(accuracy, 2),
            "precision": round(precision, 2),
            "recall": round(recall, 2),
            "f1_score": round(f1_score, 2)
        }

        return render_template("dashboard.html", metrics=metrics)

    except Exception as e:
        flash(f"Dashboard yüklenemedi: {e}", "danger")
        return redirect(url_for("index"))

@app.route('/rapor')
def rapor():
    try:
        model_date = datetime.now().strftime("%d.%m.%Y %H:%M")  # dilersen kayıtlı tarihi getir
        return render_template("report.html",
                               optimal_threshold=optimal_threshold,
                               feature_names=feature_names,
                               model_date=model_date)
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

        # ✅ Absürt değer kontrolü
        if form_data['MonthlyCharges'] > 10000 or form_data['TotalCharges'] > 1000000:
            flash("⚠️ Girdiğiniz değerler gerçek dışı görünüyor. Lütfen kontrol edin.", "danger")
            return redirect(url_for('musteri_formu'))

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


def feature_engineering(df):
    df = df.copy()
    df['TotalCharges'] = df['tenure'] * df['MonthlyCharges']

    gerekli_kolonlar = ['OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    for col in gerekli_kolonlar:
        if col not in df.columns:
            df[col] = 'No'

    df['SözleşmeSüreci'] = df['tenure'].apply(lambda x: 'Kısa' if x < 12 else ('Orta' if x < 24 else 'Uzun'))
    df['tenure_kat'] = df['tenure'].apply(lambda x: 'Kısa' if x < 12 else ('Orta' if x < 24 else 'Uzun'))
    df['monthly_kat'] = df['MonthlyCharges'].apply(lambda x: 'Düşük' if x < 35 else ('Orta' if x < 70 else 'Yüksek'))

    hizmet_kolonlari = ['PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']

    for col in ['OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']:
        if col in df.columns:
            df[col] = df[col].replace({'No internet service': 'No'})

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

def make_prediction(df):
    X_scaled = scaler.transform(df)
    prob = model.predict(X_scaled)
    pred = (prob >= optimal_threshold).astype(int)
    return pred, prob

def get_risk_level(prob):
    if prob < 0.3:
        return "Düşük"
    elif prob < 0.6:
        return "Orta"
    return "Yüksek"

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