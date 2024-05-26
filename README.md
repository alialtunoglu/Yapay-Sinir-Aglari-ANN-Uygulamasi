# Temel Yapay Sinir Ağı Uygulaması

Bu depo, Python, NumPy ve özel aktivasyon fonksiyonları kullanılarak sıfırdan oluşturulmuş temel bir yapay sinir ağı (YSA) uygulamasını içerir. Proje, veri ön işleme, model eğitimi ve Kalp Hastalığı veri seti üzerinde değerlendirme adımlarını kapsamaktadır. Bu çalışma, yapay sinir ağlarının temel yapısını anlamak ve uygulamak isteyenler için iyi bir başlangıç noktasıdır.

## Özellikler

- **Özel Yapay Sinir Ağı Sınıfı**: Katman ekleme, modeli eğitme ve tahmin yapma fonksiyonlarını içerir.
- **Özel Yoğun Katman (Dense Layer)**: Tam bağlantılı bir katmanın ileri ve geri yayılımının uygulanması.
- **Aktivasyon Fonksiyonları**: Sigmoid ve ReLU aktivasyon fonksiyonlarının uygulanması.
- **Veri Ön İşleme**: Veri setinin yüklenmesi, normalleştirilmesi ve eğitim/test setlerine bölünmesi.
- **Model Değerlendirme**: Doğruluk hesaplama ve karışıklık matrisinin görselleştirilmesi.

## Veri Seti

Modelin eğitimi ve değerlendirilmesi için Kalp Hastalığı veri seti kullanılmaktadır. Veri seti, kalp sağlığıyla ilgili çeşitli özellikler ve kalp hastalığının varlığını gösteren bir hedef değişken içermektedir.

## Başlangıç

### Gereksinimler

- Python 3.x
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

### Kurulum

1. Depoyu klonlayın:
   ```sh
   git clone https://github.com/alialtunoglu/Yapay-Sinir-Aglari-ANN-Uygulamasi.git
   cd Yapay-Sinir-Aglari-ANN-Uygulamasi
