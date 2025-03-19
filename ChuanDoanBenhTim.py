import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

pd.set_option('future.no_silent_downcasting', True)

# B1: Đọc dữ liệu
file_path = r'C:\Users\wbu86\Downloads\archive (1)\heart_disease_uci.csv'
data = pd.read_csv(file_path)

# B2: Hiển thị thông tin dữ liệu
print("Thông tin dữ liệu:")
print(data.info())
print("\nDữ liệu mẫu:")
print(data.head())

# Xử lý giá trị thiếu
for col in data.columns:
    if data[col].dtype == 'object':
        data.fillna({col: data[col].mode()[0]}, inplace=True)
    else:
        data.fillna({col: data[col].median()}, inplace=True)

data = data.infer_objects(copy=False)

# Kiểm tra lại giá trị thiếu
print("\nGiá trị thiếu sau khi xử lý:")
print(data.isnull().sum())

# B3: Mã hóa cột chuỗi thành số
data = pd.get_dummies(data)

# Kiểm tra cột mục tiêu (target)
target_column = 'target' if 'target' in data.columns else data.columns[-1]

# Vẽ heatmap chỉ với cột số
plt.figure(figsize=(12, 8))
sns.heatmap(data.corr(), annot=True, fmt=".2f", cmap='viridis', linewidths=0.5)
plt.title('Correlation Matrix of Features')
plt.show()

# Biểu đồ phân bố dữ liệu theo nhãn (target)
plt.figure(figsize=(8, 6))
sns.countplot(x=target_column, hue=target_column, data=data, palette=['#9CC9E2', '#E57373'], legend=False)
plt.title('Phân bố nhãn: 0 (Không mắc bệnh) và 1 (Mắc bệnh tim)')
plt.xlabel('0: No Disease, 1: Heart Disease')
plt.ylabel('Count')
plt.show()

# Tách đặc trưng (features) và nhãn (target)
features = data.drop(columns=target_column)
target = data[target_column]

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# B4: Chia dữ liệu thành tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.2, random_state=42)

# B5: Khởi tạo mô hình Keras
model = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(16, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

# Biên dịch mô hình
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Huấn luyện mô hình
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=1)

# B6: Đánh giá mô hình Keras
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Loss: {loss:.4f}, Accuracy: {accuracy:.4f}')

# Vẽ biểu đồ Accuracy và Loss
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

# B7: Huấn luyện và đánh giá các mô hình ML

def evaluate_model(model, model_name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f'\n🎯 {model_name} Accuracy: {acc:.4f}')
    print(classification_report(y_test, y_pred))

    plt.figure(figsize=(6, 4))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
    plt.title(f'{model_name} - Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

# 1. Logistic Regression
log_model = LogisticRegression()
evaluate_model(log_model, "Logistic Regression")

# 2. Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
evaluate_model(rf_model, "Random Forest")

# 3. XGBoost
xgb_model = xgb.XGBClassifier(eval_metric='logloss')
evaluate_model(xgb_model, "XGBoost")
