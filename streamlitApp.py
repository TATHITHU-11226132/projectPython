import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# Tiêu đề ứng dụng
st.title("Phân tích và Dự đoán PEP với Streamlit")

# 1. Load dữ liệu từ file CSV
st.header("1. Load dữ liệu từ file CSV")
try:
    print('hellohello')
    data = pd.read_csv("bank__s2.csv")
    st.write("Dữ liệu từ file bank__s2.csv:")
    st.dataframe(data.head())  # Hiển thị vài dòng đầu
except FileNotFoundError:
    st.error("Không tìm thấy file bank__s2.csv. Vui lòng đặt file trong cùng thư mục với ứng dụng!")
    st.stop()

# Tạo bản sao dữ liệu gốc để trực quan hóa
data_original = data.copy()

# 2. Thống kê dữ liệu
st.header("2. Thống kê dữ liệu")
st.write("Thống kê mô tả:")
st.write(data_original.describe())

# 3. Trực quan hóa dữ liệu (dùng dữ liệu gốc)
st.header("3. Trực quan hóa dữ liệu")

# Biểu đồ 1: Histogram - Phân phối tuổi (age)
st.subheader("Biểu đồ 1: Phân phối tuổi (age)")
fig1, ax1 = plt.subplots()
sns.histplot(data_original['age'], bins=20, kde=True, ax=ax1, color='blue')
ax1.set_title("Phân phối Tuổi")
ax1.set_xlabel("Tuổi")
ax1.set_ylabel("Số lượng")
st.pyplot(fig1)

# Biểu đồ 2: Boxplot - Phân bố thu nhập (income) theo PEP
st.subheader("Biểu đồ 2: Phân bố thu nhập theo PEP")
fig2, ax2 = plt.subplots()
sns.boxplot(x='pep', y='income', data=data_original, ax=ax2)
ax2.set_title("Phân bố Thu nhập theo PEP")
ax2.set_xlabel("PEP (0: Không, 1: Có)")
ax2.set_ylabel("Thu nhập")
st.pyplot(fig2)

# Biểu đồ 3: Heatmap - Ma trận tương quan
st.subheader("Biểu đồ 3: Ma trận tương quan")
fig3, ax3 = plt.subplots()
corr = data_original.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax3, fmt='.2f')
ax3.set_title("Ma trận Tương quan")
st.pyplot(fig3)

# Biểu đồ 4: Countplot - Số lượng theo giới tính (sex) và PEP
st.subheader("Biểu đồ 4: Số lượng theo Giới tính và PEP")
fig4, ax4 = plt.subplots()
sns.countplot(x='sex', hue='pep', data=data_original, ax=ax4)
ax4.set_title("Số lượng theo Giới tính và PEP")
ax4.set_xlabel("Giới tính (0: Nam, 1: Nữ)")
ax4.set_ylabel("Số lượng")
st.pyplot(fig4)

# Biểu đồ 5: Scatterplot - Mối quan hệ giữa tuổi và thu nhập theo PEP
st.subheader("Biểu đồ 5: Mối quan hệ giữa Tuổi và Thu nhập theo PEP")
fig5, ax5 = plt.subplots()
sns.scatterplot(x='age', y='income', hue='pep', size='children', data=data_original, ax=ax5)
ax5.set_title("Tuổi vs Thu nhập (theo PEP và số con)")
ax5.set_xlabel("Tuổi")
ax5.set_ylabel("Thu nhập")
st.pyplot(fig5)

# 4. Chuẩn hóa dữ liệu (chỉ dùng cho huấn luyện mô hình)
st.header("4. Chuẩn hóa dữ liệu (dành cho huấn luyện mô hình)")
scaler = StandardScaler()
data[['age', 'income']] = scaler.fit_transform(data[['age', 'income']])
st.write("Dữ liệu sau khi chuẩn hóa (age, income):")
st.dataframe(data.head())

# 5. Chia dữ liệu thành tập train/test
st.header("5. Chia dữ liệu thành tập train/test")
X = data.drop('pep', axis=1)
y = data['pep']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
st.write(f"Kích thước tập train: {X_train.shape}")
st.write(f"Kích thước tập test: {X_test.shape}")

# 6. Xây dựng mô hình lần 1
st.header("6. Xây dựng mô hình lần 1")
model_1 = LogisticRegression(random_state=42)
model_1.fit(X_train, y_train)
st.write("Tham số mô hình lần 1:")
st.write(model_1.coef_)

# 7. Dự đoán và đánh giá lần 1
st.header("7. Dự đoán và đánh giá lần 1")
y_pred_1 = model_1.predict(X_test)
accuracy_1 = accuracy_score(y_test, y_pred_1)
st.write(f"Độ chính xác lần 1: {accuracy_1:.2f}")

# Hiển thị báo cáo phân loại lần 1 dưới dạng bảng
st.write("Báo cáo phân loại lần 1:")
report_1 = classification_report(y_test, y_pred_1, output_dict=True)

# Tạo DataFrame từ báo cáo phân loại
report_df_1 = pd.DataFrame(report_1).transpose()
report_df_1 = report_df_1[['precision', 'recall', 'f1-score', 'support']]
report_df_1['support'] = report_df_1['support'].astype(int)  # Đảm bảo cột support là số nguyên
# Định dạng các cột số với 2 chữ số thập phân
report_df_1['precision'] = report_df_1['precision'].round(2)
report_df_1['recall'] = report_df_1['recall'].round(2)
report_df_1['f1-score'] = report_df_1['f1-score'].round(2)

# Hiển thị bảng
st.table(report_df_1)

# 8. Xây dựng mô hình lần 2 (tùy chỉnh tham số)
st.header("8. Xây dựng mô hình lần 2 (tùy chỉnh tham số)")
model_2 = LogisticRegression(random_state=42, C=0.5, max_iter=200)
model_2.fit(X_train, y_train)
st.write("Tham số mô hình lần 2:")
st.write(model_2.coef_)

# 9. Dự đoán và đánh giá lần 2
st.header("9. Dự đoán và đánh giá lần 2")
y_pred_2 = model_2.predict(X_test)
accuracy_2 = accuracy_score(y_test, y_pred_2)
st.write(f"Độ chính xác lần 2: {accuracy_2:.2f}")

# Hiển thị báo cáo phân loại lần 2 dưới dạng bảng
st.write("Báo cáo phân loại lần 2:")
report_2 = classification_report(y_test, y_pred_2, output_dict=True)

# Tạo DataFrame từ báo cáo phân loại
report_df_2 = pd.DataFrame(report_2).transpose()
report_df_2 = report_df_2[['precision', 'recall', 'f1-score', 'support']]
report_df_2['support'] = report_df_2['support'].astype(int)  # Đảm bảo cột support là số nguyên
# Định dạng các cột số với 2 chữ số thập phân
report_df_2['precision'] = report_df_2['precision'].round(2)
report_df_2['recall'] = report_df_2['recall'].round(2)
report_df_2['f1-score'] = report_df_2['f1-score'].round(2)

# Hiển thị bảng
st.table(report_df_2)

# 10. So sánh kết quả
st.header("10. So sánh kết quả")
st.write(f"Độ chính xác lần 1: {accuracy_1:.2f}")
st.write(f"Độ chính xác lần 2: {accuracy_2:.2f}")
st.write(f"Chênh lệch độ chính xác: {accuracy_2 - accuracy_1:.2f}")

# 11. Giao diện dự đoán
st.header("11. Giao diện dự đoán")
st.write("Nhập thông tin để dự đoán PEP:")
age = st.number_input("Tuổi", min_value=0, max_value=100, value=30)
sex = st.selectbox("Giới tính (0: Nam, 1: Nữ)", [0, 1])
region = st.selectbox("Khu vực (0-3)", [0, 1, 2, 3])
income = st.number_input("Thu nhập", min_value=0.0, value=20000.0)
married = st.selectbox("Tình trạng hôn nhân (0: Chưa, 1: Đã kết hôn)", [0, 1])
children = st.number_input("Số con", min_value=0, max_value=10, value=0)
car = st.selectbox("Sở hữu xe (0: Không, 1: Có)", [0, 1])
save_act = st.selectbox("Tài khoản tiết kiệm (0: Không, 1: Có)", [0, 1])
current_act = st.selectbox("Tài khoản hiện tại (0: Không, 1: Có)", [0, 1])

if st.button("Dự đoán"):
    # Chuẩn hóa dữ liệu đầu vào
    input_data = np.array([[age, sex, region, income, married, children, car, save_act, current_act]])
    input_data[:, [0, 3]] = scaler.transform(input_data[:, [0, 3]])  # Chuẩn hóa age và income
    prediction = model_2.predict(input_data)
    st.write(f"Dự đoán PEP: {'Có tham gia' if prediction[0] == 1 else 'Không tham gia'}")