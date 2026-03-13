import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.decomposition import PCA  # เพิ่ม PCA สำหรับลดมิติเพื่อวาดกราฟ

# ---------------------------------------------------------
# 1. การโหลดและการประมวลผลข้อมูลล่วงหน้า
# ---------------------------------------------------------
data = load_iris()
X = data.data
y = data.target

# ---------------------------------------------------------
# 2. การแบ่งข้อมูล (Train 60%, Validation 20%, Test 20%)
# ---------------------------------------------------------
# แบ่ง Test set ออกมาก่อน 20%
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# แบ่งส่วนที่เหลือ 80% เป็น Train (75% ของ 80% = 60%) และ Validation (25% ของ 80% = 20%)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)

# การปรับมาตราส่วนคุณลักษณะ (Feature Scaling) - สำคัญมากสำหรับ SVM
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)    # ใช้สถิติจาก Train ป้องกัน Data Leakage
X_test_scaled = scaler.transform(X_test)  # ใช้สถิติจาก Train ป้องกัน Data Leakage

# ---------------------------------------------------------
# 3. & 4. การพัฒนาแบบจำลอง และการปรับจูน Hyperparameter
# ---------------------------------------------------------
best_accuracy = 0
best_params = {}

# กำหนดค่า Hyperparameters ที่ต้องการทดลองปรับจูน
C_values = [0.1, 1, 10, 100]
kernels = ['linear', 'rbf']

print("--- ผลการปรับจูนบน Validation Set ---")
for kernel in kernels:
    for C in C_values:
        # สร้างและฝึกสอนโมเดล
        model = SVC(kernel=kernel, C=C, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # ประเมินผลบน Validation Set
        y_val_pred = model.predict(X_val_scaled)
        val_acc = accuracy_score(y_val, y_val_pred)
        
        print(f"Kernel: {kernel:<8} | C: {C:<5} | Validation Accuracy: {val_acc:.4f}")
        
        # เก็บค่าพารามิเตอร์ที่ให้ผลลัพธ์ดีที่สุด
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            best_params = {'kernel': kernel, 'C': C}

print(f"\n>> Hyperparameters ที่ดีที่สุดคือ: {best_params} (Accuracy: {best_accuracy:.4f})\n")

# ---------------------------------------------------------
# 5. การประเมินผลสุดท้ายบน Test Set ที่ไม่เคยเห็นมาก่อน
# ---------------------------------------------------------
# นำพารามิเตอร์ที่ดีที่สุดมาสร้างโมเดลตัวสุดท้าย
final_model = SVC(kernel=best_params['kernel'], C=best_params['C'], random_state=42)
final_model.fit(X_train_scaled, y_train)

# ทำนายผลบน Test Set
y_test_pred = final_model.predict(X_test_scaled)

# คำนวณ Metrics (ใช้ average='macro' เพราะมีมากกว่า 2 คลาส)
accuracy = accuracy_score(y_test, y_test_pred)
precision = precision_score(y_test, y_test_pred, average='macro')
recall = recall_score(y_test, y_test_pred, average='macro')

print("--- 5. ผลการประเมินสุดท้ายบน Test Set ---")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")


# ---------------------------------------------------------
# 6. เพิ่มส่วนการวาดกราฟ (ใช้ PCA เพื่อลดมิติข้อมูลเป็น 2D)
# ---------------------------------------------------------
print("\n--- 6. วาดกราฟแสดง Decision Boundary (ใช้ PCA) ---")

# นำข้อมูลที่ Scaled แล้วมาทำ PCA เพื่อลดเหลือ 2 มิติ
pca = PCA(n_components=2)
# Fit PCA บนข้อมูล Train (Scaled) เท่านั้น เพื่อความเสถียร
X_train_pca = pca.fit_transform(X_train_scaled)
# Transform ข้อมูล Test (Scaled) เพื่อนำมาพล็อตจุด
X_test_pca = pca.transform(X_test_scaled)

# สร้างโมเดล SVM ใหม่บนข้อมูล 2D PCA โดยใช้พารามิเตอร์ที่ดีที่สุดเดิม
plot_model = SVC(kernel=best_params['kernel'], C=best_params['C'], random_state=42)
plot_model.fit(X_train_pca, y_train)

# ฟังก์ชันสำหรับวาดกราฟ
def plot_decision_boundary(X, y, model, ax, title):
    # กำหนดขอบเขตของกราฟ
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    # สร้าง Meshgrid
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    
    # ทำนายผลทุกจุดบน Grid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # วาดพื้นหลังเป็นพื้นที่ตัดสินใจ
    ax.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    # พล็อตจุดข้อมูล
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, s=50, edgecolors='k')
    
    ax.set_title(title)
    # แสดง Legend สำหรับแต่ละคลาส
    handles, labels = scatter.legend_elements()
    class_names = data.target_names
    ax.legend(handles, class_names, title="Species", loc="best")
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')

# วาดกราฟ
fig, ax = plt.subplots(figsize=(10, 7))
title_str = f"SVM Decision Boundary (Kernel: {best_params['kernel']}, C: {best_params['C']}) on Test Data after PCA"
# พล็อตจุดข้อมูล Test ลงบนพื้นที่ตัดสินใจที่เรียนรู้จากข้อมูล Train
plot_decision_boundary(X_test_pca, y_test, plot_model, ax, title_str)
plt.tight_layout()
plt.show()