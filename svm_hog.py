import os
import time
import cv2
import numpy as np
import joblib
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# ============================================================
# 1. CẤU HÌNH & ĐƯỜNG DẪN
# ============================================================

np.random.seed(42)

base_dir = "data_split_2"
train_dir = os.path.join(base_dir, "train")
test_dir = os.path.join(base_dir, "test")

# ============================================================
# 2. HÀM AUGMENTATION NHẸ
# ============================================================

def augment_image(img):
    # Flip ngang 50% xác suất
    if np.random.rand() > 0.5:
        img = cv2.flip(img, 1)
    # Rotate ±20 độ
    angle = np.random.randint(-20, 20)
    M = cv2.getRotationMatrix2D((64,64), angle, 1)
    img = cv2.warpAffine(img, M, (128,128))
    return img

# ============================================================
# 3. HÀM LOAD ẢNH + TRÍCH XUẤT HOG + COLOR HISTOGRAM
# ============================================================

def load_data_with_features(directory, augment=False):
    features = []
    labels = []
    classes = sorted(os.listdir(directory))

    print(f"\nĐang load dữ liệu từ: {directory}")

    for label, cls in enumerate(classes):
        class_path = os.path.join(directory, cls)

        for file in os.listdir(class_path):
            img_path = os.path.join(class_path, file)
            img = cv2.imread(img_path)

            if img is None:
                continue

            img = cv2.resize(img, (128, 128))
            if augment:
                img = augment_image(img)

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # HOG feature cải thiện
            hog_feat = hog(
                gray,
                orientations=12,
                pixels_per_cell=(4, 4),
                cells_per_block=(2, 2),
                block_norm="L2-Hys"
            )

            # Color histogram (16 bins / kênh)
            hist_b = cv2.calcHist([img], [0], None, [16], [0, 256])
            hist_g = cv2.calcHist([img], [1], None, [16], [0, 256])
            hist_r = cv2.calcHist([img], [2], None, [16], [0, 256])
            color_feat = np.concatenate([hist_b, hist_g, hist_r]).flatten()

            # Kết hợp HOG + Color
            feature = np.concatenate([hog_feat, color_feat])

            features.append(feature)
            labels.append(label)

    return np.array(features), np.array(labels), classes

# ============================================================
# 4. TẠO BỘ DỮ LIỆU
# ============================================================

X_train, y_train, class_names = load_data_with_features(train_dir, augment=True)
X_test, y_test, _ = load_data_with_features(test_dir, augment=False)

print("\nClasses:", class_names)
print("Số mẫu train:", len(X_train))
print("Số mẫu test:", len(X_test))

# ============================================================
# 5. XÂY DỰNG MÔ HÌNH SVM (RBF)
# ============================================================

svm = SVC(kernel="rbf", C=10, gamma="scale", probability=True)

# ============================================================
# 6. HUẤN LUYỆN
# ============================================================

print("\nBẮT ĐẦU HUẤN LUYỆN HOG + COLOR + SVM")

start_train = time.time()
svm.fit(X_train, y_train)
end_train = time.time()
train_time = end_train - start_train

print(f"Thời gian huấn luyện: {train_time:.2f} giây")

# ============================================================
# 7. LƯU MÔ HÌNH
# ============================================================

joblib.dump(svm, "hog_svm_model_improved.pkl")
joblib.dump(class_names, "hog_svm_classes_improved.pkl")
print("\nModel đã lưu tại hog_svm_model_improved.pkl")

# ============================================================
# 8. ĐÁNH GIÁ TRÊN TEST SET
# ============================================================

print("\nĐÁNH GIÁ MÔ HÌNH")

start_eval = time.time()
y_pred = svm.predict(X_test)
end_eval = time.time()

eval_time = end_eval - start_eval
acc = accuracy_score(y_test, y_pred)

print(f"Độ chính xác trên tập test: {acc:.4f}")
print(f"Thời gian đánh giá: {eval_time:.2f} giây")

# ============================================================
# 9. BÁO CÁO CHI TIẾT
# ============================================================

print("\nBÁO CÁO CHI TIẾT\n")

report = classification_report(
    y_test,
    y_pred,
    target_names=class_names,
    digits=2
)
print(report)

# ============================================================
# 10. KẾT LUẬN
# ============================================================

print("\nTraining xong! Model HOG + COLOR + SVM đã lưu tại: hog_svm_model_improved.pkl")
