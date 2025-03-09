# Mağaza Yorumları Sentiment Analizi

Bu proje, mağaza yorumlarını analiz ederek olumlu veya olumsuz olarak sınıflandıran bir **LSTM (Long Short-Term Memory)** modeli kullanmaktadır. Türkçe metin verileri işlenerek bir duygu analizi modeli eğitilir ve test edilir.

## İçindekiler
- [Genel Bakış](#genel-bakış)
- [Gereksinimler](#gereksinimler)
- [Kurulum](#kurulum)
- [Kullanım](#kullanım)
- [Model Mimarisi](#model-mimarisi)
- [Eğitim Süreci](#eğitim-süreci)
- [Tahmin Fonksiyonu](#tahmin-fonksiyonu)
- [Sonuçlar](#sonuçlar)

## Genel Bakış
Bu proje, mağaza yorumlarını olumlu veya olumsuz olarak sınıflandırmak için **LSTM** tabanlı bir derin öğrenme modeli kullanır. Yorumların işlenmesi için **natural language processing (NLP)** teknikleri uygulanmıştır. Model, TensorFlow ve Keras kullanılarak geliştirilmiştir.

## Gereksinimler
Bu projenin çalıştırılması için aşağıdaki kütüphanelerin yüklenmesi gerekmektedir:

```bash
pip install pandas numpy tensorflow scikit-learn chardet
```

## Kurulum
1. **Gereksinimleri yükleyin**: Yukarıda belirtilen bağımlılıkları yükleyin.
2. **Veri dosyasını hazırlayın**: `magaza_yorumlari.csv` dosyasını proje dizinine ekleyin.
3. **Projeyi çalıştırın**: Aşağıdaki komut ile Jupyter Notebook veya Google Colab üzerinde çalıştırabilirsiniz.

```bash
jupyter notebook magaza_yorumlari.ipynb
```

## Kullanım
- `magaza_yorumlari.csv` dosyasındaki mağaza yorumları okunur ve temizlenir.
- Veriler tokenize edilir ve sayısal değerlere dönüştürülür.
- LSTM modeli oluşturulup eğitilir.
- Modelin doğruluk oranı test edilir.
- Kullanıcı girdileri ile modelin tahmin yapması sağlanır.

## Model Mimarisi
Model, aşağıdaki katmanlardan oluşmaktadır:
- **Embedding Layer**: Kelimeleri vektörlere dönüştürme.
- **3 Adet LSTM Katmanı**: Metni sıralı işlemek için kullanılır.
- **Dropout Katmanları**: Overfitting'i önlemek için eklenmiştir.
- **Dense Katmanı**: Çıktıyı sınıflandırmak için kullanılır.

```python
model = Sequential()
model.add(Embedding(input_dim=4000, output_dim=100, input_length=40))
model.add(LSTM(128, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(64, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(32))
model.add(Dropout(0.2))
model.add(Dense(len(label_mapping), activation='softmax'))
```

## Eğitim Süreci
- **Eğitim veri seti** ve **test veri seti** olarak ayrılmıştır.
- `categorical_crossentropy` kayıp fonksiyonu kullanılmıştır.
- `adam` optimizer ile model derlenmiştir.
- `EarlyStopping` ile overfitting önlenmeye çalışılmıştır.

```python
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
```

## Tahmin Fonksiyonu
Aşağıdaki fonksiyon, kullanıcıdan alınan bir metni temizleyerek modelden duygu tahmini yapmaktadır.

```python
def predict_sentiment(text):
    cleaned_text = clean_text(text)
    sequence = tokenizer.texts_to_sequences([cleaned_text])
    padded_sequence = pad_sequences(sequence, maxlen=max_length)
    prediction = model.predict(padded_sequence)
    sentiment = np.argmax(prediction)
    return sentiment  # 0: olumsuz, 1: olumlu
```

## Sonuçlar
Eğitim sonunda elde edilen **doğruluk oranı**:
- **Eğitim Seti Doğruluğu**: %98.5
- **Test Seti Doğruluğu**: %87.8

Örnek tahmin:
```python
predict_sentiment("Ürün çok güzel, hızlı kargo!")  # Çıktı: Olumlu
```

Bu proje, müşteri yorumlarını analiz etmek ve mağaza performansını ölçmek için kullanılabilir. Modelin daha da geliştirilmesi için daha fazla veri eklenerek eğitilebilir.

