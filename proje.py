import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, roc_curve, auc
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import time
import os

# Eğitim süresini ölçmek için başlangıç zamanını kaydedelim
start_time = time.time()

# GPU ayarları
print("TensorFlow sürümü:", tf.__version__)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"{len(gpus)} GPU bulundu ve bellek ayarları yapıldı")
    except RuntimeError as e:
        print(f"GPU bellek ayarı hatası: {e}")
else:
    print("GPU bulunamadı, CPU kullanılacak")

# Rastgelelik için seed ayarı (tekrarlanabilirlik için)
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# Çıktı dizini oluşturma
OUTPUT_DIR = 'model_ciktilari'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Veri yükleniyor...")
# 1. Veri yükleme
try:
    df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
    print("Veri dosyası başarıyla yüklendi")
except FileNotFoundError:
    try:
        df = pd.read_csv("/Users/zeynephelinaydin/proje.py/WA_Fn-UseC_-Telco-Customer-Churn.csv")
        print("Veri dosyası alternatif konumdan yüklendi")
    except FileNotFoundError:
        raise Exception("Veri dosyası bulunamadı! Lütfen doğru yolu belirtin")

# 2. Veri ön işleme ve temizleme
print("Veri ön işleme yapılıyor...")

# Veri şekli ve eksik değerlerin kontrolü
print(f"Veri seti boyutu: {df.shape}")
eksik_degerler = df.isnull().sum()
print(f"Eksik değerler:\n{eksik_degerler}")

# TotalCharges sütununu sayısal formata çevirme ve eksik değerleri doldurma
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df.apply(
    lambda row: row['MonthlyCharges'] * row['tenure'] if pd.isna(row['TotalCharges']) else row['TotalCharges'],
    axis=1
)

# Hedef değişkeni binary formata çevirme
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# Sınıf dağılımını görüntüleme
sinif_dagilimi = df['Churn'].value_counts()
print(f"Sınıf dağılımı:\n{sinif_dagilimi}")
print(f"Sınıf oranı (0/1): {sinif_dagilimi[0]/sinif_dagilimi[1]:.2f}")

# 3. Özellik mühendisliği
print("Özellik mühendisliği uygulanıyor...")

# Kullanım süresi grupları (daha az grup için q=3)
df['tenure_group'] = pd.qcut(df['tenure'], q=3, labels=['Kısa', 'Orta', 'Uzun'])
df = pd.get_dummies(df, columns=['tenure_group'], prefix='tenure', drop_first=True)

# Aylık ücret grupları (daha az grup için q=3)
df['monthly_group'] = pd.qcut(df['MonthlyCharges'], q=3, labels=['Düşük', 'Orta', 'Yüksek'])
df = pd.get_dummies(df, columns=['monthly_group'], prefix='monthly', drop_first=True)

# Önemli oranlar ve ilişkiler
df['OrtalamaSözlesmeÜcreti'] = df['MonthlyCharges'] / (df['tenure'] + 1)  # +1 sıfıra bölünmeyi önler
df['HizmetDeğeri'] = df['TotalCharges'] / (df['MonthlyCharges'] + 1)
df['SözleşmeTipi'] = df['Contract'].map({'Month-to-month': 0, 'One year': 1, 'Two year': 2})
df['SözleşmeSüresi'] = df['SözleşmeTipi'] * df['tenure']

# Hizmet sayısı
service_cols = ['PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 
                'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']

df['HizmetSayısı'] = 0
for col in service_cols:
    df['HizmetSayısı'] += df[col].apply(lambda x: 0 if x == 'No' or x == 'No phone service' or x == 'No internet service' else 1)

# Hizmet yoğunluğu
df['HizmetYoğunluğu'] = df['HizmetSayısı'] / len(service_cols)
df['HizmetBaşıMaliyet'] = df['MonthlyCharges'] / (df['HizmetSayısı'] + 1)

# 4. Kategorik değişkenleri işleme
categorical_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                   'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                   'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
                   'PaperlessBilling', 'PaymentMethod']

# One-hot encoding
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Gereksiz sütunları kaldırma
X = df_encoded.drop(columns=['customerID', 'Churn', 'SözleşmeTipi'])
y = df_encoded['Churn']

print(f"Toplam özellik sayısı: {X.shape[1]}")

# 5. Veriyi ayırma (eğitim/test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
)

# Ölçeklendirme
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. Veri dengeleme (SMOTE)
print("Veri dengeleme uygulanıyor...")
smote = SMOTE(random_state=RANDOM_SEED, sampling_strategy=0.8)  # Tam 1:1 dengeleme yerine 0.8 oranı
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

print(f"SMOTE sonrası eğitim seti dağılımı:\n{pd.Series(y_train_resampled).value_counts()}")

# 7. Model oluşturma
print("Model oluşturuluyor...")
model = Sequential([
    # Giriş katmanı - daha basit yapı için boyutu küçülttük
    Dense(64, input_shape=(X_train_resampled.shape[1],), kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    tf.keras.layers.LeakyReLU(alpha=0.1),
    Dropout(0.2),  # Dropout oranını düşürdük
    
    # Gizli katman (sayısını azalttık)
    Dense(32, kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    tf.keras.layers.LeakyReLU(alpha=0.1),
    Dropout(0.2),
    
    # Çıkış katmanı
    Dense(1, activation='sigmoid')
])

# Model özeti
model.summary()

# 8. Model derleme - daha yüksek öğrenme oranı
optimizer = Adam(learning_rate=0.005)  # Öğrenme oranını artırdık
model.compile(
    optimizer=optimizer,
    loss='binary_crossentropy',
    metrics=['accuracy', 'Precision', 'Recall', 'AUC']
)

# 9. Erken durdurma callback'i - daha düşük patience değeri
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=7,  # Daha düşük sabır değeri
    restore_best_weights=True,
    verbose=1
)

# 10. Model eğitimi - validation oranını azalttık, batch size arttırdık
print("Model eğitiliyor...")
history = model.fit(
    X_train_resampled,
    y_train_resampled,
    validation_split=0.15,  # Validation oranını azalttık
    epochs=50,              # Epoch sayısını azalttık
    batch_size=128,         # Batch size arttırdık (daha hızlı eğitim)
    callbacks=[early_stopping],
    verbose=1,
    class_weight={0: 1, 1: 1}  # Sınıf ağırlıklarını eşit yaptık
)

# 11. Eğitim grafiklerini oluşturma
print("Eğitim grafikleri çiziliyor...")
plt.figure(figsize=(15, 10))

# Loss grafiği
plt.subplot(2, 2, 1)
plt.plot(history.history['loss'], label='Eğitim Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Accuracy grafiği
plt.subplot(2, 2, 2)
plt.plot(history.history['accuracy'], label='Eğitim Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Precision ve Recall grafiği
plt.subplot(2, 2, 3)
plt.plot(history.history['Precision'], label='Eğitim Precision')
plt.plot(history.history['val_Precision'], label='Validation Precision')
plt.plot(history.history['Recall'], label='Eğitim Recall')
plt.plot(history.history['val_Recall'], label='Validation Recall')
plt.title('Precision ve Recall')
plt.xlabel('Epoch')
plt.ylabel('Değer')
plt.legend()

# 12. Test veri seti üzerinde tahmin yapma
print("Test veri seti üzerinde tahmin yapılıyor...")
y_pred_proba = model.predict(X_test_scaled, verbose=0)

# 13. ROC eğrisi çizme
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.subplot(2, 2, 4)
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/egitim_grafikleri.png')

# 14. En iyi eşik değerini bulma
print("En iyi eşik değeri belirleniyor...")
# Eşik değerleri için precision-recall değerlerini hesaplama
precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_proba)

# Her eşik değeri için F1 skorlarını hesaplama
f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-8)
optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_idx]

print(f"En iyi eşik değeri: {optimal_threshold:.3f}")

# 15. Precision-Recall eğrisi
plt.figure(figsize=(10, 8))
plt.plot(recalls[:-1], precisions[:-1], 'b-', label='Precision-Recall eğrisi')
plt.axvline(x=recalls[optimal_idx], color='r', linestyle='--', 
            label=f'En iyi eşik: {optimal_threshold:.3f}')
plt.title('Precision-Recall Eğrisi')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.grid(True)
plt.legend()
plt.savefig(f'{OUTPUT_DIR}/precision_recall_curve.png')

# 16. En iyi eşik değeriyle tahmin
y_pred = (y_pred_proba >= optimal_threshold).astype(int)

# 17. Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Kalacak (0)', 'Ayrılacak (1)'],
            yticklabels=['Kalacak (0)', 'Ayrılacak (1)'])
plt.title('Karışıklık Matrisi (Confusion Matrix)')
plt.xlabel('Tahmin Edilen')
plt.ylabel('Gerçek')
plt.savefig(f'{OUTPUT_DIR}/confusion_matrix.png')

# 18. Sınıflandırma raporu
print("\nOptimize Edilmiş Sınıflandırma Raporu:")
classification_rep = classification_report(y_test, y_pred, target_names=['Kalacak', 'Ayrılacak'])
print(classification_rep)

# 19. Özellik önem sıralaması
# Bu kısmı model türü nedeniyle ekleyemiyoruz ama bir açıklama olarak bırakıyoruz
print("\nNot: Yapay sinir ağları özellik önemini doğrudan göstermez. Özellik önemliliğini belirlemek için")
print("permütasyon yöntemleri veya gradyan tabanlı yöntemler kullanılabilir ancak bu işlem yoğun hesaplama")
print("gerektirir ve bu kodda eklenmemiştir.")

# 20. Modeli ve bileşenleri kaydetme
print("\nModel ve bileşenler kaydediliyor...")
model.save(f'{OUTPUT_DIR}/musteri_kaybi_modeli.h5')
joblib.dump(scaler, f'{OUTPUT_DIR}/olceklendirici.pkl')
joblib.dump(optimal_threshold, f'{OUTPUT_DIR}/optimal_esik.pkl')
joblib.dump(X.columns.tolist(), f'{OUTPUT_DIR}/ozellik_isimleri.pkl')

# Sınıflandırma raporunu dosyaya kaydet
with open(f'{OUTPUT_DIR}/siniflandirma_raporu.txt', 'w') as f:
    f.write(classification_rep)

# 21. Tahmin fonksiyonu
def musteri_kaybi_tahmin(veri, esik_degeri=None, model_dizini=OUTPUT_DIR):
    """
    Yeni müşteri verileri için kayıp tahmini yapar
    
    Parametreler:
    veri: DataFrame - tahmin yapılacak müşteri verisi
    esik_degeri: float - özel bir eşik değeri (belirtilmezse optimize edilmiş değeri kullanır)
    model_dizini: str - model dosyalarının bulunduğu dizin
    
    Dönüş değerleri:
    tahminler: array - binary tahminler (0: Kalacak, 1: Ayrılacak)
    olasiliklar: array - Ayrılma olasılıkları
    """
    
    # Gerekli dosyaları yükleme
    from tensorflow.keras.models import load_model
    import joblib
    
    try:
        model = load_model(f'{model_dizini}/musteri_kaybi_modeli.h5')
        scaler = joblib.load(f'{model_dizini}/olceklendirici.pkl')
        ozellik_isimleri = joblib.load(f'{model_dizini}/ozellik_isimleri.pkl')
        
        if esik_degeri is None:
            esik_degeri = joblib.load(f'{model_dizini}/optimal_esik.pkl')
    except Exception as e:
        raise Exception(f"Model dosyaları yüklenirken hata oluştu: {e}")
    
    # Veri özelliklerini kontrol et
    eksik_ozellikler = [col for col in ozellik_isimleri if col not in veri.columns]
    if eksik_ozellikler:
        print(f"Uyarı: Giriş verisinde eksik özellikler var: {eksik_ozellikler}")
    
    # Veriyi yeniden düzenle
    veri = veri.reindex(columns=ozellik_isimleri, fill_value=0)
    
    # Veriyi ölçeklendir
    X_scaled = scaler.transform(veri)
    
    # Tahmin yap
    olasiliklar = model.predict(X_scaled, verbose=0)
    tahminler = (olasiliklar >= esik_degeri).astype(int)
    
    return tahminler, olasiliklar

# 22. Toplam çalışma süresini hesapla ve göster
end_time = time.time()
duration = end_time - start_time
print(f"\nToplam çalışma süresi: {duration:.2f} saniye ({duration/60:.2f} dakika)")

print("\nİşlem tamamlandı! Model ve sonuçlar '{}' dizinine kaydedildi.".format(OUTPUT_DIR))

# Örnek kod - tahmin fonksiyonunun kullanımı
print("\nTahmin fonksiyonu örnek kullanımı:")
print("tahminler, olasiliklar = musteri_kaybi_tahmin(yeni_musteri_verisi)")