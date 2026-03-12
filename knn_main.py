import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

# 1. data preprocessing
#---------------------------------------------------
iris = load_iris(as_frame=True)
X = iris.data  # ใช้ features ทั้งหมด
y = iris.target

# 2. การแบ่งข้อมูล(data splitting) 
#---------------------------------------------------
# แบ่งเป็น (Train + Validation) 80% และ Test 20%
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# แบ่ง X_temp เป็น Train 75% และ Validation 25%(ของ X_temp) 
# สัดส่วนสุดท้ายจะเป็น Train 60%, Val 20%, Test 20%
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42)

# feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# 3. และ 4. modal และ การปรับจูน Hyperparameter
#---------------------------------------------------
# หาค่า K ที่ดีที่สุดโดยใช้ชุด Validation set
k_values = range(1, 16)
best_k = 1
best_val_accuracy = 0

print("--- Hyperparameter Tuning (Validation Set) ---")
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    
    # ประเมินบน Validation Set
    val_preds = knn.predict(X_val_scaled)
    val_acc = accuracy_score(y_val, val_preds)
    
    print(f"K = {k:2d} | Validation Accuracy: {val_acc:.4f}")
    
    if val_acc > best_val_accuracy:
        best_val_accuracy = val_acc
        best_k = k

print(f"\n>> Best K found: {best_k} (Validation Accuracy: {best_val_accuracy:.4f})\n")

# 5. การประเมินผลสุดท้ายบน Test Set 
#---------------------------------------------------
# สร้างโมเดลสุดท้ายด้วยค่า K ที่ดีที่สุด และรวม Train + Validation เข้าด้วยกันเพื่อประสิทธิภาพสูงสุด
X_train_final = np.vstack((X_train_scaled, X_val_scaled))
y_train_final = np.concatenate((y_train, y_val))

final_knn = KNeighborsClassifier(n_neighbors=best_k)
final_knn.fit(X_train_final, y_train_final)

# ทดสอบบน Test Set ที่ไม่เคยเห็น 
y_test_preds = final_knn.predict(X_test_scaled)

# คำนวณ metrics  
# (ใช้ average='macro' เพราะเป็นข้อมูล Multiclass)
test_accuracy = accuracy_score(y_test, y_test_preds)
test_precision = precision_score(y_test, y_test_preds, average='macro')
test_recall = recall_score(y_test, y_test_preds, average='macro')

print("--- Final Evaluation (Test Set) ---")
print(f"Accuracy : {test_accuracy:.4f}")
print(f"Precision: {test_precision:.4f}")
print(f"Recall   : {test_recall:.4f}")