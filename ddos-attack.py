import time
import os
import pandas as pd 
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics,svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,precision_score
from sklearn.metrics import recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


raw_df = pd.read_csv("/kaggle/input/ddos-saldirisi/proje2_veri_seti.txt", delimiter=',')
raw_df.head()


raw_df[' Label: Benign'].unique()


def convert_label(label):
    if label == ' Label: Benign':
        return 'Benign'
    else:
        return 'DDoS'

# Etiket dönüşümünü gerçekleştirme
raw_df['balance_Label'] = raw_df[' Label: Benign'].apply(convert_label)

# ' Label: Benign' sütununu kaldırma
raw_df.drop(columns=[' Label: Benign'], inplace=True)

# Yeni sütunun adını ' Label: Benign' olarak değiştirme
raw_df.rename(columns={'balance_Label': ' Label: Benign'}, inplace=True)

raw_df.head()



def handle_non_numerical_data(df):
    columns = df.columns.values
    for column in columns:
        text_digit_vals = {}
        def convert_to_int(val):
            return text_digit_vals[val]

        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            column_contents = df[column].values.tolist()
            unique_elements = set(column_contents)
            x = 0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x+=1

            df[column] = list(map(convert_to_int, df[column]))

    return df


clean_df=handle_non_numerical_data(raw_df)


clean_df.head()


sns.histplot(data=raw_df, x=' Label: Benign', hue=' Label: Benign', multiple='stack')
plt.show()


print(clean_df.columns)

feature_column = ['IP Source: 10.0.2.15',' IP Destination: 192.168.1.102',' TCP Source Port: 1415',' TCP Destination Port: 445',' IP Protocol: 6',' Frame Length: 174',' TCP Flags (SYN): True',' TCP Flags (RST): False',' TCP Flags (PUSH): False',' TCP Flags (ACK): False',' TCP Sequence Number: 935361675',' TCP ACK Number: 470528611',' Frame Time: 1714745712.019767',' Label: Benign']
X = clean_df[feature_column] 
Y = clean_df[' Label: Benign']  # 'balance_Label' yerine 'Label: Benign' kullanıldı
print(X.shape)
print(Y.shape)


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42) 

print(X_train.shape)
print(y_train.shape)

print(X_test.shape)
print(y_test.shape)


print(X_test.shape[0])


# Decision Tree Classifier modelini oluşturma ve eğitme
start_time = time.time() # Başlangıç zamanını kaydet
decision_tree_model = DecisionTreeClassifier()
decision_tree_model.fit(X_train, y_train)
training_time = time.time() - start_time # Eğitim süresini hesapla

# Test veri seti üzerinde tahmin yapma
start_time = time.time() # Başlangıç zamanını kaydet
y_pred_decision_tree = decision_tree_model.predict(X_test)
prediction_time = time.time() - start_time # Tahmin süresini hesapla

# Accuracy değerini hesaplama
accuracy = accuracy_score(y_test, y_pred_decision_tree)
print("Accuracy:", accuracy)

# Precision değerini hesaplama
precision = precision_score(y_test, y_pred_decision_tree, pos_label=0)
print("Precision:", precision)

# Recall değerini hesaplama
recall = recall_score(y_test, y_pred_decision_tree, pos_label=0)
print("Recall:", recall)

# F1 score değerini hesaplama
f1 = f1_score(y_test, y_pred_decision_tree, pos_label=0)
print("F1 Score:", f1)

# Confusion matrix çizimi
cm = confusion_matrix(y_test, y_pred_decision_tree)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=['Predicted 0', 'Predicted 1'],
            yticklabels=['Actual 0', 'Actual 1'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

print("Training Time:", training_time) # Eğitim süresini yazdır
print("Prediction Time:", prediction_time) # Tahmin süresini yazdır




# Random Forest Classifier modelini oluşturma ve eğitme
start_time = time.time() # Başlangıç zamanını kaydet
random_forest_model = RandomForestClassifier()
random_forest_model.fit(X_train, y_train)
training_time_rf = time.time() - start_time # Eğitim süresini hesapla

# Test veri seti üzerinde tahmin yapma
start_time = time.time() # Başlangıç zamanını kaydet
y_pred_random_forest = random_forest_model.predict(X_test)
prediction_time_rf = time.time() - start_time # Tahmin süresini hesapla

# Accuracy değerini hesaplama
accuracy_rf = accuracy_score(y_test, y_pred_random_forest)
print("Accuracy:", accuracy_rf)

# Precision değerini hesaplama
precision_rf = precision_score(y_test, y_pred_random_forest, pos_label=0)
print("Precision:", precision_rf)

# Recall değerini hesaplama
recall_rf = recall_score(y_test, y_pred_random_forest, pos_label=0)
print("Recall:", recall_rf)

# F1 score değerini hesaplama
f1_rf = f1_score(y_test, y_pred_random_forest, pos_label=0)
print("F1 Score:", f1_rf)

# Confusion matrix çizimi
cm_rf = confusion_matrix(y_test, y_pred_random_forest)
sns.heatmap(cm_rf, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=['Predicted 0', 'Predicted 1'],
            yticklabels=['Actual 0', 'Actual 1'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Random Forest')
plt.show()

print("Training Time (Random Forest):", training_time_rf) # Eğitim süresini yazdır
print("Prediction Time (Random Forest):", prediction_time_rf) # Tahmin süresini yazdır



# KNeighbors Classifier modelini oluşturma ve eğitme
start_time = time.time() # Başlangıç zamanını kaydet
knn_model = KNeighborsClassifier()
knn_model.fit(X_train, y_train)
training_time_knn = time.time() - start_time # Eğitim süresini hesapla

# Test veri seti üzerinde tahmin yapma
start_time = time.time() # Başlangıç zamanını kaydet
y_pred_knn = knn_model.predict(X_test)
prediction_time_knn = time.time() - start_time # Tahmin süresini hesapla

# Accuracy değerini hesaplama
accuracy_knn = accuracy_score(y_test, y_pred_knn)
print("Accuracy:", accuracy_knn)

# Precision değerini hesaplama
precision_knn = precision_score(y_test, y_pred_knn, pos_label=0)
print("Precision:", precision_knn)

# Recall değerini hesaplama
recall_knn = recall_score(y_test, y_pred_knn, pos_label=0)
print("Recall:", recall_knn)

# F1 score değerini hesaplama
f1_knn = f1_score(y_test, y_pred_knn, pos_label=0)
print("F1 Score:", f1_knn)

# Confusion matrix çizimi
cm_knn = confusion_matrix(y_test, y_pred_knn)
sns.heatmap(cm_knn, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=['Predicted 0', 'Predicted 1'],
            yticklabels=['Actual 0', 'Actual 1'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - KNeighbors')
plt.show()

print("Training Time (KNeighbors):", training_time_knn) # Eğitim süresini yazdır
print("Prediction Time (KNeighbors):", prediction_time_knn) # Tahmin süresini yazdır



# SVM modelini oluşturma ve eğitme
start_time = time.time() # Başlangıç zamanını kaydet
svm_model = svm.SVC()
svm_model.fit(X_train, y_train)
training_time_svm = time.time() - start_time # Eğitim süresini hesapla

# Test veri seti üzerinde tahmin yapma
start_time = time.time() # Başlangıç zamanını kaydet
y_pred_svm = svm_model.predict(X_test)
prediction_time_svm = time.time() - start_time # Tahmin süresini hesapla

# Accuracy değerini hesaplama
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print("Accuracy:", accuracy_svm)

# Precision değerini hesaplama
precision_svm = precision_score(y_test, y_pred_svm, pos_label=0)
print("Precision:", precision_svm)

# Recall değerini hesaplama
recall_svm = recall_score(y_test, y_pred_svm, pos_label=0)
print("Recall:", recall_svm)

# F1 score değerini hesaplama
f1_svm = f1_score(y_test, y_pred_svm, pos_label=0)
print("F1 Score:", f1_svm)

# Confusion matrix çizimi
cm_svm = confusion_matrix(y_test, y_pred_svm)
sns.heatmap(cm_svm, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=['Predicted 0', 'Predicted 1'],
            yticklabels=['Actual 0', 'Actual 1'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - SVM')
plt.show()

print("Training Time (SVM):", training_time_svm) # Eğitim süresini yazdır
print("Prediction Time (SVM):", prediction_time_svm) # Tahmin süresini yazdır



# Yapay Sinir Ağı (Neural Network) modelini oluşturma ve eğitme
start_time = time.time() # Başlangıç zamanını kaydet
nn_model = MLPClassifier(random_state=1, max_iter=300)
nn_model.fit(X_train, y_train)
training_time_nn = time.time() - start_time # Eğitim süresini hesapla

# Test veri seti üzerinde tahmin yapma
start_time = time.time() # Başlangıç zamanını kaydet
y_pred_nn = nn_model.predict(X_test)
prediction_time_nn = time.time() - start_time # Tahmin süresini hesapla

# Accuracy değerini hesaplama
accuracy_nn = accuracy_score(y_test, y_pred_nn)
print("Accuracy:", accuracy_nn)

# Precision değerini hesaplama
precision_nn = precision_score(y_test, y_pred_nn, pos_label=0)
print("Precision:", precision_nn)

# Recall değerini hesaplama
recall_nn = recall_score(y_test, y_pred_nn, pos_label=0)
print("Recall:", recall_nn)

# F1 score değerini hesaplama
f1_nn = f1_score(y_test, y_pred_nn, pos_label=0)
print("F1 Score:", f1_nn)

# Confusion matrix çizimi
cm_nn = confusion_matrix(y_test, y_pred_nn)
sns.heatmap(cm_nn, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=['Predicted 0', 'Predicted 1'],
            yticklabels=['Actual 0', 'Actual 1'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Neural Network')
plt.show()

print("Training Time (Neural Network):", training_time_nn) # Eğitim süresini yazdır
print("Prediction Time (Neural Network):", prediction_time_nn) # Tahmin süresini yazdır



# Gradient Boosting Tree modelini oluşturma ve eğitme
start_time = time.time() # Başlangıç zamanını kaydet
gbt_model = GradientBoostingClassifier(random_state=1)
gbt_model.fit(X_train, y_train)
training_time_gbt = time.time() - start_time # Eğitim süresini hesapla

# Test veri seti üzerinde tahmin yapma
start_time = time.time() # Başlangıç zamanını kaydet
y_pred_gbt = gbt_model.predict(X_test)
prediction_time_gbt = time.time() - start_time # Tahmin süresini hesapla

# Accuracy değerini hesaplama
accuracy_gbt = accuracy_score(y_test, y_pred_gbt)
print("Accuracy:", accuracy_gbt)

# Precision değerini hesaplama
precision_gbt = precision_score(y_test, y_pred_gbt, pos_label=0)
print("Precision:", precision_gbt)

# Recall değerini hesaplama
recall_gbt = recall_score(y_test, y_pred_gbt, pos_label=0)
print("Recall:", recall_gbt)

# F1 score değerini hesaplama
f1_gbt = f1_score(y_test, y_pred_gbt, pos_label=0)
print("F1 Score:", f1_gbt)

# Confusion matrix çizimi
cm_gbt = confusion_matrix(y_test, y_pred_gbt)
sns.heatmap(cm_gbt, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=['Predicted 0', 'Predicted 1'],
            yticklabels=['Actual 0', 'Actual 1'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Gradient Boosting Tree')
plt.show()

print("Training Time (Gradient Boosting Tree):", training_time_gbt) # Eğitim süresini yazdır
print("Prediction Time (Gradient Boosting Tree):", prediction_time_gbt) # Tahmin süresini yazdır



# Logistic Regression modelini oluşturma ve eğitme
start_time = time.time() # Başlangıç zamanını kaydet
log_reg_model = LogisticRegression(random_state=1)
log_reg_model.fit(X_train, y_train)
training_time_log_reg = time.time() - start_time # Eğitim süresini hesapla

# Test veri seti üzerinde tahmin yapma
start_time = time.time() # Başlangıç zamanını kaydet
y_pred_log_reg = log_reg_model.predict(X_test)
prediction_time_log_reg = time.time() - start_time # Tahmin süresini hesapla

# Accuracy değerini hesaplama
accuracy_log_reg = accuracy_score(y_test, y_pred_log_reg)
print("Accuracy:", accuracy_log_reg)

# Precision değerini hesaplama
precision_log_reg = precision_score(y_test, y_pred_log_reg, pos_label=0)
print("Precision:", precision_log_reg)

# Recall değerini hesaplama
recall_log_reg = recall_score(y_test, y_pred_log_reg, pos_label=0)
print("Recall:", recall_log_reg)

# F1 score değerini hesaplama
f1_log_reg = f1_score(y_test, y_pred_log_reg, pos_label=0)
print("F1 Score:", f1_log_reg)

# Confusion matrix çizimi
cm_log_reg = confusion_matrix(y_test, y_pred_log_reg)
sns.heatmap(cm_log_reg, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=['Predicted 0', 'Predicted 1'],
            yticklabels=['Actual 0', 'Actual 1'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Logistic Regression')
plt.show()

print("Training Time (Logistic Regression):", training_time_log_reg) # Eğitim süresini yazdır
print("Prediction Time (Logistic Regression):", prediction_time_log_reg) # Tahmin süresini yazdır


# Doğruluk değerlerini saklamak için bir sözlük oluşturma
accuracy_dict = {}

# Her bir model için doğruluk değerlerini hesaplayıp sözlüğe eklemek
models = ['Decision Tree', 'KNeighbors', 'RandomForest', 'Support Vector Machine', 'Neural Network', 'Gradient Boosting', 'Logistic Regression']
for model_name in models:
    model = None
    if model_name == 'Decision Tree':
        model = DecisionTreeClassifier(random_state=1)
    elif model_name == 'KNeighbors':
        model = KNeighborsClassifier()
    elif model_name == 'RandomForest':
        model = RandomForestClassifier(random_state=1)
    elif model_name == 'Support Vector Machine':
        model = svm.SVC()
    elif model_name == 'Neural Network':
        model = MLPClassifier(random_state=1)
    elif model_name == 'Gradient Boosting':
        model = GradientBoostingClassifier(random_state=1)
    elif model_name == 'Logistic Regression':
        model = LogisticRegression(random_state=1)
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_dict[model_name] = accuracy

# Doğruluk değerlerini sütun grafiği olarak görselleştirme
plt.figure(figsize=(10, 6))
plt.bar(accuracy_dict.keys(), accuracy_dict.values(), color='skyblue')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.title('Accuracy of Different Models')
plt.xticks(rotation=45)
plt.ylim(0, 1)  # Y ekseni 0 ile 1 arasında olacak şekilde sınırlanır
plt.show()



# Precision değerlerini saklamak için bir sözlük oluşturma
precision_dict = {}

# Her bir model için precision değerlerini hesaplayıp sözlüğe eklemek
models = ['Decision Tree', 'KNeighbors', 'RandomForest', 'Support Vector Machine', 'Neural Network', 'Gradient Boosting', 'Logistic Regression']
for model_name in models:
    model = None
    if model_name == 'Decision Tree':
        model = DecisionTreeClassifier(random_state=1)
    elif model_name == 'KNeighbors':
        model = KNeighborsClassifier()
    elif model_name == 'RandomForest':
        model = RandomForestClassifier(random_state=1)
    elif model_name == 'Support Vector Machine':
        model = svm.SVC()
    elif model_name == 'Neural Network':
        model = MLPClassifier(random_state=1)
    elif model_name == 'Gradient Boosting':
        model = GradientBoostingClassifier(random_state=1)
    elif model_name == 'Logistic Regression':
        model = LogisticRegression(random_state=1)
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Etiketleri sayısal değerlere dönüştürme
    label_encoder = LabelEncoder()
    y_test_encoded = label_encoder.fit_transform(y_test)
    precision = precision_score(y_test_encoded, y_pred, pos_label=1)  # 'Benign' etiketini 1 olarak kabul ediyoruz
    precision_dict[model_name] = precision

# Precision değerlerini sütun grafiği olarak görselleştirme
plt.figure(figsize=(10, 6))
plt.bar(precision_dict.keys(), precision_dict.values(), color='lightgreen')
plt.xlabel('Model')
plt.ylabel('Precision')
plt.title('Precision of Different Models')
plt.xticks(rotation=45)
plt.ylim(0, 1)  # Y ekseni 0 ile 1 arasında olacak şekilde sınırlanır
plt.show()



# Recall değerlerini saklamak için bir sözlük oluşturma
recall_dict = {}

# Her bir model için recall değerlerini hesaplayıp sözlüğe eklemek
models = ['Decision Tree', 'KNeighbors', 'RandomForest', 'Support Vector Machine', 'Neural Network', 'Gradient Boosting', 'Logistic Regression']
for model_name in models:
    model = None
    if model_name == 'Decision Tree':
        model = DecisionTreeClassifier(random_state=1)
    elif model_name == 'KNeighbors':
        model = KNeighborsClassifier()
    elif model_name == 'RandomForest':
        model = RandomForestClassifier(random_state=1)
    elif model_name == 'Support Vector Machine':
        model = svm.SVC()
    elif model_name == 'Neural Network':
        model = MLPClassifier(random_state=1)
    elif model_name == 'Gradient Boosting':
        model = GradientBoostingClassifier(random_state=1)
    elif model_name == 'Logistic Regression':
        model = LogisticRegression(random_state=1)
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    recall = recall_score(y_test, y_pred, pos_label=0)  # 'Benign' etiketini 0 ile değiştirdik
    recall_dict[model_name] = recall

# Recall değerlerini sütun grafiği olarak görselleştirme
plt.figure(figsize=(10, 6))
plt.bar(recall_dict.keys(), recall_dict.values(), color='lightblue')
plt.xlabel('Model')
plt.ylabel('Recall')
plt.title('Recall of Different Models')
plt.xticks(rotation=45)
plt.ylim(0, 1)  # Y ekseni 0 ile 1 arasında olacak şekilde sınırlanır
plt.show()



# F1 skorlarını saklamak için bir sözlük oluşturma
f1_score_dict = {}

# Her bir model için F1 skorlarını hesaplayıp sözlüğe eklemek
models = ['Decision Tree', 'KNeighbors', 'RandomForest', 'Support Vector Machine', 'Neural Network', 'Gradient Boosting', 'Logistic Regression']
for model_name in models:
    model = None
    if model_name == 'Decision Tree':
        model = DecisionTreeClassifier(random_state=1)
    elif model_name == 'KNeighbors':
        model = KNeighborsClassifier()
    elif model_name == 'RandomForest':
        model = RandomForestClassifier(random_state=1)
    elif model_name == 'Support Vector Machine':
        model = svm.SVC()
    elif model_name == 'Neural Network':
        model = MLPClassifier(random_state=1)
    elif model_name == 'Gradient Boosting':
        model = GradientBoostingClassifier(random_state=1)
    elif model_name == 'Logistic Regression':
        model = LogisticRegression(random_state=1)
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    f1_score_val = f1_score(y_test, y_pred, pos_label=0)  # 'Benign' etiketini 0 ile değiştirdik
    f1_score_dict[model_name] = f1_score_val

# F1 skorlarını sütun grafiği olarak görselleştirme
plt.figure(figsize=(10, 6))
plt.bar(f1_score_dict.keys(), f1_score_dict.values(), color='lightcoral')
plt.xlabel('Model')
plt.ylabel('F1 Score')
plt.title('F1 Score of Different Models')
plt.xticks(rotation=45)
plt.ylim(0, 1)  # Y ekseni 0 ile 1 arasında olacak şekilde sınırlanır
plt.show()




# Model isimleri ve eğitim sürelerini saklamak için sözlük oluşturma
training_time_dict = {}

# Her bir model için eğitim sürelerini hesaplayıp sözlüğe eklemek
models = ['Decision Tree', 'KNeighbors', 'RandomForest', 'Support Vector Machine', 'Neural Network', 'Gradient Boosting', 'Logistic Regression']
for model_name in models:
    model = None
    if model_name == 'Decision Tree':
        model = DecisionTreeClassifier(random_state=1)
    elif model_name == 'KNeighbors':
        model = KNeighborsClassifier()
    elif model_name == 'RandomForest':
        model = RandomForestClassifier(random_state=1)
    elif model_name == 'Support Vector Machine':
        model = svm.SVC()
    elif model_name == 'Neural Network':
        model = MLPClassifier(random_state=1)
    elif model_name == 'Gradient Boosting':
        model = GradientBoostingClassifier(random_state=1)
    elif model_name == 'Logistic Regression':
        model = LogisticRegression(random_state=1)
    
    start_time = time.time()  # Eğitim başlangıç zamanını kaydet
    model.fit(X_train, y_train)
    training_time = time.time() - start_time  # Eğitim süresini hesapla
    training_time_dict[model_name] = training_time

# Eğitim sürelerini çizgi grafiği olarak görselleştirme
plt.figure(figsize=(10, 6))
plt.plot(training_time_dict.keys(), training_time_dict.values(), marker='o', color='orange', linestyle='-')
plt.xlabel('Model')
plt.ylabel('Training Time (seconds)')
plt.title('Training Time of Different Models')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()



# Tahmin sürelerini saklamak için sözlük oluşturma
prediction_time_dict = {}

# Her bir model için tahmin sürelerini hesaplayıp sözlüğe eklemek
models = ['Decision Tree', 'KNeighbors', 'RandomForest', 'Support Vector Machine', 'Neural Network', 'Gradient Boosting', 'Logistic Regression']
for model_name in models:
    model = None
    if model_name == 'Decision Tree':
        model = DecisionTreeClassifier(random_state=1)
    elif model_name == 'KNeighbors':
        model = KNeighborsClassifier()
    elif model_name == 'RandomForest':
        model = RandomForestClassifier(random_state=1)
    elif model_name == 'Support Vector Machine':
        model = svm.SVC()
    elif model_name == 'Neural Network':
        model = MLPClassifier(random_state=1)
    elif model_name == 'Gradient Boosting':
        model = GradientBoostingClassifier(random_state=1)
    elif model_name == 'Logistic Regression':
        model = LogisticRegression(random_state=1)
    
    model.fit(X_train, y_train)  # Modeli eğit
    
    # Tahmin süresini ölçmek için önceden belirlenmiş test verisi üzerinde tahmin yap
    start_time = time.time()  # Tahmin başlangıç zamanını kaydet
    model.predict(X_test)
    prediction_time = time.time() - start_time  # Tahmin süresini hesapla
    prediction_time_dict[model_name] = prediction_time

# Tahmin sürelerini çizgi grafiği olarak görselleştirme
plt.figure(figsize=(10, 6))
plt.plot(prediction_time_dict.keys(), prediction_time_dict.values(), marker='o', color='purple', linestyle='-')
plt.xlabel('Model')
plt.ylabel('Prediction Time (seconds)')
plt.title('Prediction Time of Different Models')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()
