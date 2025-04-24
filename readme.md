
---

### 📌 README.md

# ✋ El Hareketi ile Parlaklık Kontrol Uygulaması

Bu proje, MediaPipe kullanarak el hareketleriyle ekran parlaklığını kontrol etmenizi sağlar. Başparmak ve işaret parmağı arasındaki mesafeye göre ekran parlaklığı ayarlanır. Ayrıca FPS bilgisi ve mesafe değeri ekranda gösterilir.

---

## 🧰 Kullanılan Kütüphaneler

Bu projede aşağıdaki Python kütüphaneleri kullanılmıştır:

- `opencv-python` → Kameradan görüntü almak ve görsel çıktı göstermek için.
- `mediapipe` → El hareketlerini algılamak için.
- `numpy` → Sayısal işlemler için.
- `screen-brightness-control` → Ekran parlaklığını kontrol etmek için.
- `math` ve `time` → Mesafe ve FPS hesaplamak için.

---

## 💻 Kurulum

Projeyi çalıştırmadan önce aşağıdaki adımları takip ederek bağımlılıkları kurmanız gerekir:

1. Proje klasöründe `requirements.txt` dosyası bulunmaktadır.
2. Aşağıdaki komutla tüm kütüphaneleri kurabilirsiniz:

bash
pip install -r requirements.txt


> Not: `screen-brightness-control` sadece Windows işletim sisteminde çalışır. Linux veya macOS'ta farklı yöntemler gerekebilir.

---

## 🚀 Kullanım

1. `hand_landmarker.task` adlı model dosyasını proje dizinine ekleyin.
2. Aşağıdaki komutla uygulamayı başlatın:

bash
python app.py


> Varsayılan olarak bilgisayarın kamerası açılır. Elinizi kameraya göstererek başparmak ve işaret parmağını birbirine yaklaştırıp uzaklaştırarak ekran parlaklığını değiştirebilirsiniz.

---

## 🔧 Yapılan Değişiklikler

- `set_brightness_based_on_distance()` fonksiyonu eklendi: Bu fonksiyon, el hareketlerine göre parlaklık ayarlaması yapar.
- FPS hesaplama eklendi ve ekranda gösterildi.
- `draw_landmarks_on_image()` fonksiyonuna el yönü bilgisi (left/right) çizimi eklendi.
- `screen_brightness_control` paketi projeye dahil edildi.
- Hatalar için try-except bloklarıyla kapsamlı hata yönetimi sağlandı.

---

## 📝 Notlar

- Proje MediaPipe'in **HandLandmarker** modeli ile çalışmaktadır. `.task` dosyası gereklidir.
- `cv2.VideoCapture(0)` kullanılarak varsayılan kamera açılır, isterseniz `1`, `2` gibi değerlerle farklı kameraları seçebilirsiniz.

---

## 👨‍💻 Geliştirici

Burak Arda Işık
Bilgisayar Programcılığı – 2. Sınıf  


---
