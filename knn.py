import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

file_path = 'C:\\Área de Trabalho\\Base\\iris.txt'  # Altere para o seu arquivo


df = pd.read_csv(file_path, delimiter=',', header=None)

df.columns = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)', 'species']

print("Primeiras linhas do dataset:")
print(df.head())

X = df[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']]
y = df['species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

knn = KNeighborsClassifier(n_neighbors=3) 
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'Acurácia do modelo: {accuracy:.2f}')

plt.figure(figsize=(10, 6))
scatter = plt.scatter(
    X['sepal length (cm)'],
    X['sepal width (cm)'],
    c=y.map({'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}),
    cmap='viridis',
    edgecolor='k',
    s=100
)

plt.title('Gráfico de Dispersão da Íris')
plt.xlabel('Comprimento da Sépala (cm)')
plt.ylabel('Largura da Sépala (cm)')

legend_labels = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
handles = [plt.Line2D([0], [0], marker='o', color='w', label=label, 
                       markerfacecolor=scatter.cmap(i/2), markersize=10) 
           for label, i in legend_labels.items()]
plt.legend(handles=handles, title="Classes")

plt.grid(True)
plt.show()
