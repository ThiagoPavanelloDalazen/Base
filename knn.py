import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

file_path = r"C:\Users\VonTh\OneDrive\Área de Trabalho\Base\iris.txt" #Atenção trocar local onde está o .txt
df = pd.read_csv(file_path, delimiter=',', header=None)


X = df.iloc[:, :-1].values 
y = df.iloc[:, -1].values   


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)


knn_cv_scores = cross_val_score(knn, X, y, cv=5)
knn_accuracy = knn_cv_scores.mean()


y_pred_knn = knn.predict(X_test)


knn_accuracy_score = accuracy_score(y_test, y_pred_knn)
knn_precision = precision_score(y_test, y_pred_knn, average='weighted')
knn_recall = recall_score(y_test, y_pred_knn, average='weighted')
knn_f1 = f1_score(y_test, y_pred_knn, average='weighted')


svm = SVC()
svm.fit(X_train, y_train)


svm_cv_scores = cross_val_score(svm, X, y, cv=5)
svm_accuracy = svm_cv_scores.mean()


y_pred_svm = svm.predict(X_test)


svm_accuracy_score = accuracy_score(y_test, y_pred_svm)
svm_precision = precision_score(y_test, y_pred_svm, average='weighted')
svm_recall = recall_score(y_test, y_pred_svm, average='weighted')
svm_f1 = f1_score(y_test, y_pred_svm, average='weighted')


print("Resultados para KNN:")
print(f"Acurácia (Validação Cruzada): {knn_accuracy:.2f}")
print(f"Acurácia (Teste): {knn_accuracy_score:.2f}")
print(f"Precisão: {knn_precision:.2f}")
print(f"Revocação: {knn_recall:.2f}")
print(f"F1-score: {knn_f1:.2f}")

print("\nResultados para SVM:")
print(f"Acurácia (Validação Cruzada): {svm_accuracy:.2f}")
print(f"Acurácia (Teste): {svm_accuracy_score:.2f}")
print(f"Precisão: {svm_precision:.2f}")
print(f"Revocação: {svm_recall:.2f}")
print(f"F1-score: {svm_f1:.2f}")


print("\nMatriz de Confusão para KNN:")
print(confusion_matrix(y_test, y_pred_knn))

print("\nMatriz de Confusão para SVM:")
print(confusion_matrix(y_test, y_pred_svm))
