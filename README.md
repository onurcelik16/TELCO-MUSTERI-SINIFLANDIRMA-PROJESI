# Yapay Sinir Ağı ile Müşteri Kaybı (Churn) Tahmini

Bu proje, müşteri verilerine dayanarak bir müşterinin hizmeti bırakıp bırakmayacağını tahmin etmek amacıyla yapay sinir ağı (ANN) modeli kullanılarak geliştirilmiştir. Model, veri ön işleme, dengesiz veri seti için SMOTE uygulaması ve değerlendirme metrikleriyle desteklenmiştir.

## 🔍 Proje Amacı

- 📉 Müşteri kaybını önceden tahmin etmek
- 📊 Kurumların sadakat stratejilerine katkı sağlamak
- 🧠 Derin öğrenme tabanlı çözüm sunmak

## 🚀 Özellikler

- 📚 Veri temizleme ve ön işleme
- ⚖️ SMOTE ile veri dengeleme
- 🧠 Keras ile ANN modeli oluşturma
- 🧪 Accuracy, Precision, Recall, F1-score gibi metriklerle model değerlendirme
- 📈 Confusion Matrix ve ROC eğrisi ile performans analizi
- 🌐 Flask tabanlı web arayüz (isteğe bağlı)
- 📁 Tahmin fonksiyonu ile kullanıcı girdisinden anlık tahmin alma

## 🧱 Kullanılan Teknolojiler

- Python
- Pandas, NumPy
- Scikit-learn
- Keras (TensorFlow backend)
- Matplotlib, Seaborn
- imblearn (SMOTE)
- Flask (opsiyonel web arayüz için)

## 📁 Dosya Yapısı

📦ann-churn-prediction
┣ 📂models
┃ ┗ model.h5
┣ 📂static / 📂templates (Flask varsa)
┣ 📜main.py (Model eğitimi)
┣ 📜predict.py (Tahmin fonksiyonu)
┣ 📜app.py (Flask sunucusu)
┣ 📜churn_data.csv
┗ 📜README.md
## 🛠️ Kurulum ve Çalıştırma

```bash
git clone https://github.com/kullaniciadi/ann-churn-prediction.git
cd ann-churn-prediction
pip install -r requirements.txt
python main.py  # modeli eğitir
python app.py   # Flask arayüzü başlatır (isteğe bağlı)
📊 Model Başarımı (Örnek)
Accuracy: %80

Precision: %81

Recall: %82

F1-score: %81

Gerçek sonuçlar model_evaluation.txt içinde detaylı olarak sunulmuştur.

---

📝 Elinizde **proje ekran görüntüsü**, **demo linki** veya **model görseli** varsa bana gönderin, README’ye ekleyelim.

İstersen bir de Türkçe + İngilizce çift dil destekli yapabiliriz.
