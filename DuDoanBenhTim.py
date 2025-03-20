import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

# Đọc dữ liệu
file_path = r'C:\Users\wbu86\Downloads\archive (1)\heart_disease_uci.csv'
data = pd.read_csv(file_path)

# Hiển thị các cột trong dữ liệu để kiểm tra
print("Các cột có trong file CSV:")
print(data.columns.tolist())

# Xác định cột target chính xác
target_col = 'target' if 'target' in data.columns else 'output' if 'output' in data.columns else 'num' if 'num' in data.columns else None
if target_col is None:
    raise ValueError("Không tìm thấy cột 'target', 'output' hoặc 'num' trong file CSV")

# Chuyển đổi giá trị mục tiêu thành nhị phân (0: không bệnh, 1: có bệnh)
y = (data[target_col] > 0).astype(int)

# Tách feature
X = data.drop([target_col, 'id'], axis=1, errors='ignore')

# Kiểm tra và in ra các cột có kiểu dữ liệu không phải số
non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
print("Các cột không phải số:", non_numeric_cols)

# Xử lý các cột không phải số bằng One-Hot Encoding nếu có
if non_numeric_cols:
    X = pd.get_dummies(X, columns=non_numeric_cols)

# Kiểm tra lại dữ liệu sau khi mã hóa
print("Dữ liệu sau khi mã hóa:")
print(X.head())

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Chia train/test
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42)

# Xây dựng model mạng nơ-ron
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))  # Dự đoán nhị phân

# Compile model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Huấn luyện model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                    epochs=100, batch_size=16, verbose=1)

# Đánh giá model
y_pred = (model.predict(X_test) > 0.5).astype("int32")
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Lưu model
model.save("heart_disease_prediction_model.h5")
print("\n✅ Model đã được lưu thành công.")
