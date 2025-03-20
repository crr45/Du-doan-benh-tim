import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
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
y = (data[target_col] > 0).astype(int) #Chuyển các giá trị dương thành True, các giá trị 0 thành False.
#Chuyển đổi kiểu dữ liệu từ True/False thành 1/0.


# Tách feature
X = data.drop([target_col, 'id'], axis=1, errors='ignore') #drop(): Loại bỏ cột mục tiêu và cột 'id' (nếu có) để lấy các cột đặc trưng.

# Kiểm tra và in ra các cột có kiểu dữ liệu không phải số
non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
print("Các cột không phải số:", non_numeric_cols)

# Xử lý các cột không phải số 
if non_numeric_cols:#Lọc ra các cột không phải kiểu số.
    X = pd.get_dummies(X, columns=non_numeric_cols) #Biến đổi các cột dạng chuỗi thành dạng One-Hot Encoding.

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
model = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')  # Dự đoán nhị phân
])

'''
Input(shape=(X_train.shape[1],)): Lớp đầu vào với số đặc trưng của dữ liệu.
Dense(64, activation='relu'): Lớp ẩn với 64 neuron và hàm kích hoạt ReLU.
Dropout(0.3): Xác suất 30% vô hiệu hóa ngẫu nhiên neuron để tránh overfitting.
Dense(1, activation='sigmoid'): Lớp đầu ra với 1 neuron dùng hàm sigmoid để phân loại nhị phân.
'''


# Compile model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
'''
optimizer='adam': Sử dụng thuật toán tối ưu hóa Adam.
loss='binary_crossentropy': Hàm mất mát cho bài toán phân loại nhị phân.
metrics=['accuracy']: Đánh giá mô hình dựa trên độ chính xác.
'''

# Huấn luyện model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                    epochs=100, batch_size=16, verbose=1)

'''
epochs=100: Số lần lặp huấn luyện toàn bộ dữ liệu.
batch_size=16: Kích thước mỗi lô dữ liệu.
validation_data: Sử dụng tập kiểm tra để đánh giá mô hình sau mỗi epoch.
verbose=1: Hiển thị chi tiết quá trình huấn luyện.
'''


# Đánh giá model
y_pred = (model.predict(X_test) > 0.5).astype("int32")
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=1))

'''
model.predict(): Dự đoán kết quả trên tập kiểm tra.
accuracy_score(): Tính độ chính xác.
classification_report(): Hiển thị các chỉ số đánh giá như Precision, Recall, F1-score.
'''

# Lưu model
model.save("heart_disease_prediction_model.keras")
print("\n✅ Model đã được lưu thành công.")
