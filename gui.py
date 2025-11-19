import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
import tensorflow as tf
import os
import joblib
from skimage.feature import hog

# ============================================================
# 1. LOAD MODELS
# ============================================================
model = tf.keras.models.load_model("fruit_model.h5")
train_dir = "data_split_2/train"
class_names = sorted(os.listdir(train_dir))
svm_model = joblib.load("hog_svm_model_improved.pkl")
svm_classes = joblib.load("hog_svm_classes_improved.pkl")

# ============================================================
# 2. PREDICTION FUNCTIONS
# ============================================================
def predict_image(img: Image.Image):
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    preds = model.predict(img_array, verbose=0)
    class_index = np.argmax(preds)
    confidence = float(np.max(preds)) * 100
    return class_names[class_index], confidence

def predict_hog_svm(img: Image.Image):
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    img_resized = cv2.resize(img_cv, (128, 128))
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

    hog_feat = hog(gray, orientations=12, pixels_per_cell=(4,4),
                   cells_per_block=(2,2), block_norm="L2-Hys")

    hist_r = cv2.calcHist([img_resized], [0], None, [16], [0,256])
    hist_g = cv2.calcHist([img_resized], [1], None, [16], [0,256])
    hist_b = cv2.calcHist([img_resized], [2], None, [16], [0,256])
    color_feat = np.concatenate([hist_r, hist_g, hist_b]).flatten()

    features = np.concatenate([hog_feat, color_feat]).reshape(1, -1)

    pred_label = svm_model.predict(features)[0]
    prob = svm_model.predict_proba(features)
    confidence = float(np.max(prob)) * 100

    return svm_classes[pred_label], confidence


# ============================================================
# 3. TKINTER GUI - GIAO DIỆN ĐẸP, HIỆN ĐẠI
# ============================================================
root = tk.Tk()
root.title("Hệ thống nhận dạng trái cây")
root.geometry("1100x700")
root.configure(bg="#f0f2f5")

# ================= TITLE ==================
title_label = tk.Label(
    root,
    text="HỆ THỐNG NHẬN DẠNG TRÁI CÂY",
    font=("Segoe UI", 26, "bold"),
    bg="#f0f2f5",
    fg="#2c3e50"
)
title_label.pack(pady=25)

# ================= MAIN FRAME (CARD) ==================
main_frame = tk.Frame(root, bg="#ffffff")
main_frame.pack(expand=True, fill="both", padx=40, pady=20)
main_frame.configure(highlightbackground="#d0d0d0", highlightthickness=1)

# ================= LEFT PANEL ==================
left_frame = tk.Frame(main_frame, bg="#ffffff")
left_frame.grid(row=0, column=0, sticky="ns", padx=(25, 50), pady=20)

result_label = tk.Label(
    left_frame,
    text="MobileNet-V2:\n---",
    font=("Segoe UI", 15, "bold"),
    bg="#ffffff",
    fg="#007acc",
    justify="left"
)
result_label.pack(pady=(0,20), anchor="w")

svm_label = tk.Label(
    left_frame,
    text="HOG-SVM:\n---",
    font=("Segoe UI", 15, "bold"),
    bg="#ffffff",
    fg="#8e44ad",
    justify="left"
)
svm_label.pack(pady=(0,30), anchor="w")

# ========= Style button =========
def style_button(btn, bg, hover):
    def on_enter(e): btn["bg"] = hover
    def on_leave(e): btn["bg"] = bg
    btn.bind("<Enter>", on_enter)
    btn.bind("<Leave>", on_leave)

def create_button(text, color, hover, cmd):
    btn = tk.Button(
        left_frame,
        text=text,
        command=cmd,
        fg="white",
        bg=color,
        activebackground=hover,
        font=("Segoe UI", 13, "bold"),
        width=18,
        height=2,
        bd=0,
        relief="flat",
    )
    btn.pack(pady=10)
    style_button(btn, color, hover)
    return btn

btn_upload = create_button("Tải ảnh lên", "#4CAF50", "#45a049", lambda: upload_image())
btn_clear  = create_button("Xóa ảnh", "#f39c12", "#e67e22", lambda: clear_image())

# ================= RIGHT PANEL ==================
right_frame = tk.Frame(main_frame, bg="#ffffff")
right_frame.grid(row=0, column=1, sticky="nsew", pady=20)
main_frame.grid_columnconfigure(1, weight=1)
main_frame.grid_rowconfigure(0, weight=1)

# ====== Canvas hiển thị ảnh ======
image_frame = tk.Frame(right_frame, bg="#e6e6e6")
image_frame.pack(expand=True, fill="both", padx=20, pady=10)
image_frame.configure(highlightbackground="#c0c0c0", highlightthickness=1)

canvas = tk.Label(
    image_frame,
    bg="#d9d9d9",
    text="Chưa có ảnh",
    font=("Segoe UI", 16),
    fg="#555",
)
canvas.pack(expand=True, fill="both", padx=20, pady=20)


# ============================================================
# 4. FUNCTIONS
# ============================================================
def upload_image():
    file_path = filedialog.askopenfilename(
        filetypes=[("Image Files", "*.jpg;*.png;*.jpeg")]
    )
    if not file_path:
        return

    img = Image.open(file_path).convert("RGB")

    result_cnn, conf_cnn = predict_image(img)
    result_svm, conf_svm = predict_hog_svm(img)

    result_label.config(text=f"MobileNet-V2:\n{result_cnn} ({conf_cnn:.2f}%)")
    svm_label.config(text=f"HOG-SVM:\n{result_svm} ({conf_svm:.2f}%)")

    img_resized = img.resize((500, 450))
    tk_img = ImageTk.PhotoImage(img_resized)

    canvas.config(image=tk_img, text="")
    canvas.image = tk_img

def clear_image():
    result_label.config(text="MobileNet-V2:\n---")
    svm_label.config(text="HOG-SVM:\n---")
    canvas.config(image="", text="Chưa có ảnh")


# ============================================================
# 5. RUN GUI
# ============================================================
root.mainloop()
