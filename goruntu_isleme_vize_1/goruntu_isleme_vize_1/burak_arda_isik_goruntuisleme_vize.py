import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import cv2
import screen_brightness_control as sbc
import math
import time # FPS hesaplamak için

MARGIN = 10
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54)
BRIGHTNESS_INFO_COLOR = (255, 0, 0)
FPS_COLOR = (0, 255, 0)

# Parlaklık Kontrol Fonksiyonu
def set_brightness_based_on_distance(image, single_hand_landmarks):
    """
    Tek bir elin başparmak ve işaret parmağı uçları arasındaki mesafeyi hesaplar
    ve ekran parlaklığını buna göre ayarlar.
    """
    calculated_brightness = None
    distance = None
    # Başparmak (4) ve işaret parmağı (8) için yeterli parmak izinin olup olmadığını kontrol et
    if single_hand_landmarks and len(single_hand_landmarks) > 8: # En az 9 parmak izi gerekiyor (0-8)
        try:
            thumb_tip = single_hand_landmarks[4]
            index_finger_tip = single_hand_landmarks[8]
            # Görüntü boyutlarını al (piksel koordinatları için)
            h, w, _ = image.shape

            # Parmak uçlarının piksel koordinatlarını al
            thumb_x, thumb_y = int(thumb_tip.x * w), int(thumb_tip.y * h)
            index_x, index_y = int(index_finger_tip.x * w), int(index_finger_tip.y * h)

            # Öklid mesafesini hesapla
            distance = math.hypot(thumb_x - index_x, thumb_y - index_y)

            # Mesafeyi normalize et
            min_distance = 0   # Parmaklar dokunduğunda beklenen minimum piksel mesafesi
            max_distance = 100  # Parmaklar birbirinden uzakken beklenen maksimum piksel mesafesi
            # max_distance'in min_distance'ten büyük olduğundan emin ol (sıfıra bölme hatasını önle)
            if max_distance <= min_distance:
                max_distance = min_distance + 100 # Eğer max çok küçükse varsayılan bir aralık ata

            # np.clip ile değeri 0 ve 1 arasında sınırla
            normalized_distance = np.clip((distance - min_distance) / (max_distance - min_distance), 0, 1)

            # Normalize mesafeyi parlaklık yüzdesine (0-100) dönüştür
            calculated_brightness = int(normalized_distance * 100)

            # Ekran parlaklığını ayarla
            sbc.set_brightness(calculated_brightness, display=0) 

        except IndexError:
              print("Hata: Landmark indeksi sınırların dışında.")
              return None, None
        except Exception as e:
            print(f"Parlaklık ayarlanamadı: {e}")
            # Ayarlama başarısız olursa parlaklık için None döndür
            return None, distance

    # Hesaplanan parlaklığı ve piksel mesafesini döndür
    return calculated_brightness, distance

# Landmark Çizim Fonksiyonu
def draw_landmarks_on_image(rgb_image, detection_result):
    # El işaretlerini ve el yönünü görüntü üzerine çiz.
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness
    annotated_image = np.copy(rgb_image)
    height, width, _ = annotated_image.shape # Görüntü boyutlarını bir kez al

    # Algılanan elleri görselleştirmek için döngü.
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx] # Mevcut el için işaretler
        handedness = handedness_list[idx] # Mevcut el için yön bilgisi

        # MediaPipe çizim yardımcılarını kullanarak el işaretlerini çiz.
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z) for lm in hand_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            hand_landmarks_proto,
            solutions.hands.HAND_CONNECTIONS,
            solutions.drawing_styles.get_default_hand_landmarks_style(),
            solutions.drawing_styles.get_default_hand_connections_style())

        # El yönü metnini çizme
        try:
            # Metnin konumunu hesaplamak için landmark koordinatlarını kullan
            x_coordinates = [landmark.x for landmark in hand_landmarks]
            y_coordinates = [landmark.y for landmark in hand_landmarks]
            if not x_coordinates or not y_coordinates:
                  continue # Koordinat yoksa atla

            text_x = int(min(x_coordinates) * width)
            text_y = int(min(y_coordinates) * height) - MARGIN

            # El yönünü (sol veya sağ) çiz.
            if handedness and handedness[0].category_name:
                  cv2.putText(annotated_image, f"{handedness[0].category_name}",
                              (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                              FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)
        except Exception as e:
            print(f"El yönü metni çizilirken hata: {e}")

    return annotated_image


# Ana Program

# HandLandmarker nesnesi oluştur.
model_path = 'hand_landmarker.task'
try:
    print("HandLandmarker modeli yükleniyor...")
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO, # VIDEO modu kullanılmalı
        num_hands=2)
    detector = vision.HandLandmarker.create_from_options(options)
    print("Model başarıyla yüklendi.")
except Exception as e:
    print(f"Model yüklenirken veya '{model_path}' dosyasından detector oluşturulurken hata: {e}")
    exit()

# Video yakalamayı başlat.
print("Kamera başlatılıyor...")
cam = cv2.VideoCapture(0) 
if not cam.isOpened():
    print("Hata: Kamera açılamadı.")
    if detector: detector.close() # Dedektör oluşturulduysa kapat
    exit()
print("Kamera başlatıldı.")

# FPS hesaplaması için değişkenler
prev_time = 0
fps = 0

# Ana döngü: Video karelerini işle.
print("Video akışı işlenmeye başlanıyor...")
while True:
    # Kameradan bir kare oku
    success, frame = cam.read()
    if not success:
        print("Boş kamera karesi yoksayılıyor.")
        continue

    # Görüntü boyutlarını al 
    height, width, _ = frame.shape

    # FPS hesapla
    current_time = time.time()
    if prev_time > 0:
        fps = 1 / (current_time - prev_time)
    prev_time = current_time

    # BGR görüntüyü RGB'ye dönüştür 
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # MediaPipe Image nesnesi oluştur.
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # El işaretlerini algıla.
    frame_timestamp_ms = int(current_time * 1000)
    detection_result = detector.detect_for_video(mp_image, frame_timestamp_ms)

    # Çizim için orijinal BGR karesiyle başla
    annotated_image = frame

    # Parlaklık Kontrolü & Landmark Çizimi 
    brightness_value = None
    distance_value = None
    if detection_result.hand_landmarks:
        # Parlaklık kontrolü için ilk algılanan eli kullan
        first_hand_landmarks = detection_result.hand_landmarks[0]
        # Parlaklığı hesapla 
        brightness_value, distance_value = set_brightness_based_on_distance(rgb_frame, first_hand_landmarks)

        # İşaretleri RGB kare üzerine çiz
        annotated_image_rgb = draw_landmarks_on_image(rgb_frame, detection_result)
        # Çizilmiş RGB görüntüyü tekrar BGR'ye dönüştür
        annotated_image = cv2.cvtColor(annotated_image_rgb, cv2.COLOR_RGB2BGR)

    # FPS bilgisini kare üzerine yazdır
    cv2.putText(annotated_image, f"FPS: {int(fps)}", (width - 100, 30), # Tanımlanmış genişliği kullan
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, FPS_COLOR, 2)

    # Parlaklık bilgisi varsa yazdır
    if brightness_value is not None:
        # Mesafe piksel cinsinden döndürülüyor bu fonksiyonda
        cv2.putText(annotated_image, f"Mesafe: {int(distance_value) if distance_value is not None else 'N/A'}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, BRIGHTNESS_INFO_COLOR, 2)
        cv2.putText(annotated_image, f"Parlaklik: {brightness_value}%",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, BRIGHTNESS_INFO_COLOR, 2)

    # İşlenmiş kareyi göster.
    cv2.imshow("Hand Landmark Brightness Control", annotated_image)

    # 'q' tuşuna basılırsa döngüden çık.
    if cv2.waitKey(5) & 0xFF == ord('q'):
        print("Çıkış tuşuna basıldı. Kaynaklar serbest bırakılıyor...")
        break

# Kaynakları serbest bırak.
cam.release()
cv2.destroyAllWindows()
if detector: detector.close() # Dedektör oluşturulduysa kapat
print("Kamera serbest bırakıldı, pencereler kapatıldı ve dedektör kapatıldı.")