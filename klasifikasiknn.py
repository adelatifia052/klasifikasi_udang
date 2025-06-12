import os
import cv2
import numpy as np
from collections import Counter
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

# === Konfigurasi ===
IMG_SIZE = (100, 100)
CLASSES = ["segar", "tidak_segar"]
K = 3
DATASET_DIR = "dataset"

# Ekstrak fitur HSV
def extract_hsv_features(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, IMG_SIZE)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    return [np.mean(h), np.mean(s), np.mean(v)]

# Load data training
def load_dataset():
    features, labels = [], []
    for label in CLASSES:
        folder = os.path.join(DATASET_DIR, label)
        if not os.path.exists(folder): continue
        for fname in os.listdir(folder):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                try:
                    path = os.path.join(folder, fname)
                    feats = extract_hsv_features(path)
                    features.append(feats)
                    labels.append(label)
                except: continue
    return np.array(features), np.array(labels)

# KNN sederhana
def knn_predict(test_point, X_train, y_train, k=3):
    distances = np.linalg.norm(X_train - test_point, axis=1)
    nearest = np.argsort(distances)[:k]
    nearest_labels = y_train[nearest]
    return Counter(nearest_labels).most_common(1)[0][0]

# Histogram HSV sebagai gambar
def create_histogram_image(hsv_img):
    hist_size, hist_h, hist_w = 256, 150, 256
    single_hist_canvas = np.zeros((hist_h, hist_w, 3), dtype=np.uint8)
    histograms = []

    for i, color in enumerate([(0, 0, 255), (0, 255, 0), (255, 0, 0)]):  # H, S, V
        hist = cv2.calcHist([hsv_img], [i], None, [hist_size], [0, 256])
        cv2.normalize(hist, hist, 0, hist_h, cv2.NORM_MINMAX)
        canvas = single_hist_canvas.copy()
        for x in range(1, hist_size):
            y1, y2 = int(hist[x - 1]), int(hist[x])
            cv2.line(canvas, (x - 1, hist_h - y1), (x, hist_h - y2), color, 1)
        histograms.append(canvas)

    # Gabungkan horizontal: H | S | V
    combined = cv2.hconcat(histograms)

    # Tambahkan label teks di bawah
    label_height = 30
    labeled_canvas = np.full((hist_h + label_height, combined.shape[1], 3), 255, dtype=np.uint8)
    labeled_canvas[:hist_h] = combined

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale, thickness = 0.6, 1
    labels = ['Hue', 'Saturation', 'Value']
    for i in range(3):
        x_pos = i * hist_w + 10
        cv2.putText(labeled_canvas, labels[i], (x_pos, hist_h + 20), font, font_scale, (0, 0, 0), thickness)

    return labeled_canvas


# Tampilkan histogram di GUI
def show_histogram(hsv_img):
    hist_img = create_histogram_image(hsv_img)
    hist_img_rgb = cv2.cvtColor(hist_img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(hist_img_rgb)
    img_tk = ImageTk.PhotoImage(img_pil)
    panel_hist.config(image=img_tk)
    panel_hist.image = img_tk

# Klasifikasi dan tampilkan hasil
def classify_image():
    path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
    if not path: return
    try:
        img_cv = cv2.imread(path)
        img_cv = cv2.resize(img_cv, IMG_SIZE)
        hsv_img = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
        feats = extract_hsv_features(path)
        pred = knn_predict(feats, X_train, y_train, K)

        # Penjelasan berdasarkan prediksi
        if pred == "segar":
            penjelasan = "Warna dan kecerahan menunjukkan ciri udang segar."
        else:
            penjelasan = "Warna dan kecerahan rendah mengindikasikan udang tidak segar."

        # Tampilkan gambar udang
        img_disp = cv2.cvtColor(cv2.resize(cv2.imread(path), (300, 300)), cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_disp)
        img_tk = ImageTk.PhotoImage(img_pil)
        panel_image.config(image=img_tk)
        panel_image.image = img_tk

        label_pred.config(text=f"Hasil: {pred}")
        label_explain.config(text=penjelasan)

        show_histogram(hsv_img)
    except Exception as e:
        messagebox.showerror("Error", f"Gagal mengklasifikasi: {e}")

# ==== Load data training ====
X_train, y_train = load_dataset()

# ==== GUI ====
root = tk.Tk()
root.title("Klasifikasi Kesegaran Udang (HSV + KNN)")
root.geometry("800x600")
root.configure(bg="#083c6d")

tk.Label(root, text="🦐 Klasifikasi Kesegaran Udang", font=("Segoe UI", 18, "bold"),
         fg="white", bg="#083c6d").pack(pady=10)

tk.Button(root, text="📷 Pilih Gambar", font=("Segoe UI", 14),
          command=classify_image, bg="#145DA0", fg="white").pack(pady=10)

panel_image = tk.Label(root, bg="#083c6d")
panel_image.pack(pady=10)

label_pred = tk.Label(root, text="Belum ada prediksi", font=("Segoe UI", 14), fg="white", bg="#083c6d")
label_pred.pack()

label_explain = tk.Label(root, text="", font=("Segoe UI", 12), fg="white", bg="#083c6d", wraplength=600)
label_explain.pack(pady=5)

panel_hist = tk.Label(root, bg="#083c6d")
panel_hist.pack(pady=10)

root.mainloop()
