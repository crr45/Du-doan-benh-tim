import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Đọc dữ liệu
data = pd.read_csv('/content/drive/MyDrive/Bài tập lớn nhóm 6/Colab Notebooks/heart-disease.csv')

# Tách dữ liệu đầu vào và đầu ra
X = data.drop(columns='target', axis=1)
Y = data['target']

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Tách dữ liệu thành tập huấn luyện và kiểm tra
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=42)

# Xây dựng mô hình mạng neuron
model = Sequential([
    Dense(16, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Biên dịch mô hình
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Huấn luyện mô hình
model.fit(X_train, Y_train, epochs=100, batch_size=16, verbose=1)

# Đánh giá mô hình
loss, accuracy = model.evaluate(X_test, Y_test)
print(f'✅ Độ chính xác trên tập kiểm tra: {accuracy:.2f}')

# Hàm dự đoán kèm tên người
def du_doan_ten(ten, input_data):
    input_df = pd.DataFrame([input_data], columns=X.columns)
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0][0]

    print(f"\n👉 Dự đoán cho {ten}:")
    print(f"   Xác suất mắc bệnh tim: {prediction:.2f}")
    if prediction > 0.5:
        print("   → Có nguy cơ bị bệnh tim")
    else:
        print("   → Không bị bệnh tim")

# Dự đoán cho từng người
du_doan_ten("Nguyễn Văn A", (67,1,0,150,280,0,0,1000,1,3.6,0,2,2))
du_doan_ten("Trần Thị B", (58,0,1,120,250,0,1,140,0,1.2,1,0,2))
du_doan_ten("Nguyễn Văn C", (22,1,0,110,120,0,0,120,0,0.0,0,0,0))

