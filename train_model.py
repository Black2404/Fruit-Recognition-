# train_model_finetune.py
import os
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import classification_report

# 1. CẤU HÌNH VÀ ĐƯỜNG DẪN

np.random.seed(42)
tf.random.set_seed(42)

base_dir = "data_split_2"
train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "val")
test_dir = os.path.join(base_dir, "test")

# 2. DATA AUGMENTATION

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=45,
    width_shift_range=0.4,
    height_shift_range=0.4,
    zoom_range=[0.5, 2.0],
    brightness_range=[0.2, 2.0],
    shear_range=0.4,
    channel_shift_range=50,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode="nearest"
)

val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# 3. TẠO CÁC BỘ DỮ LIỆU

#resize ảnh đầu vào -> 224x224 phù hợp với MobileNetV2, set 1 lần học 32 ảnh và tạo nhãn đầu ra dạng one hot vector
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical"
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical"
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
    #không làm xáo trộn thứ tự ảnh khi test
    shuffle=False
)

num_classes = len(train_generator.class_indices)
print("Classes:", train_generator.class_indices)

# 4. XÂY DỰNG MÔ HÌNH
#tải trọng số của MobileNetV2 đã được huấn luyện trên tập ImageNet: kh cần dạy model từ đầu -> Transfer Learning.
    #đã được học các đặc trung chung
    # mobilenet-v2 có phần top 1000 lớp -> bỏ đi vì chỉ có 11 lớp và tự xây lại qua Dense
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

# Freeze 50 lớp đầu, fine-tune phần sau
for layer in base_model.layers[:50]:
    layer.trainable = False
for layer in base_model.layers[50:]:
    layer.trainable = True
# sau output thì mỗi ảnh sẽ ở dạng (batch_size,H, W, C) -> (số ảnh xử lý/1 lần, kênh màu RGB)
x = base_model.output
# chuyển từ dạng 4D sau CNN rút đặc trung -> 2D (batch_size, C) tức mỗi ảnh từ 3D -> 1D
x = GlobalAveragePooling2D()(x)
# chuẩn hóa vecto đặc trưng
x = BatchNormalization()(x)
# giảm 50% neuron bất kì trong 1 batch -> giảm overfitting (kh học quá kĩ 1 số neuron cố định), giúp học đặc trung tổng quát
x = Dropout(0.5)(x)
# tính loss và kích hoạt sofmax -> ra xác suất cho classification
predictions = Dense(num_classes, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=predictions)

# 5. COMPILE MÔ HÌNH

model.compile(
    #thuật toán tối ưu Adam học với trọng số chậm, tính sai số khi học, đo tỉ lệ đoán đúng
    optimizer=Adam(learning_rate=1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# 6. CALLBACKS

early_stop = EarlyStopping(
# quan sát loss trên validation, val_loss kh giảm trong 5 epoch -> stop train -> load model với val_loss thấp nhất
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
    #khi callback đc gọi thì hiện thông báo train như nào
    verbose=1
)

checkpoint = ModelCheckpoint(
    "fruit_model.h5",
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)

lr_reduce = ReduceLROnPlateau(

    monitor="val_loss",
#một lần nữa cho mô hình học chậm hơn để tinh chỉnh trọng số x0.3
    factor=0.3,
    patience=3,
    min_lr=1e-7,
    #khi learning-rate giảm thì hiện tbao
    verbose=1
)

# 7. HUẤN LUYỆN

print("\nBẮT ĐẦU HUẤN LUYỆN")
start_train = time.time()

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=50,
    callbacks=[early_stop, checkpoint, lr_reduce],
    verbose=1
)

end_train = time.time()
train_time = end_train - start_train
print(f"\nThời gian huấn luyện: {train_time:.2f} giây")

# 8. ĐÁNH GIÁ TRÊN TEST SET

print("\nĐÁNH GIÁ MÔ HÌNH")
start_eval = time.time()
test_loss, test_acc = model.evaluate(test_generator, verbose=1)
end_eval = time.time()
eval_time = end_eval - start_eval

print(f"Độ chính xác trên tập test: {test_acc:.4f}")
print(f"Thời gian đánh giá: {eval_time:.2f} giây")

# 9. BÁO CÁO CHI TIẾT

print("\nBÁO CÁO CHI TIẾT")
y_true = test_generator.classes
y_pred_prob = model.predict(test_generator, verbose=1)
y_pred = np.argmax(y_pred_prob, axis=1)

class_labels = list(test_generator.class_indices.keys())

report = classification_report(
    y_true,
    y_pred,
    target_names=class_labels,
    digits=2
)
print(report)

print("\nTraining xong! Model đã lưu tại: fruit_model.h5")
