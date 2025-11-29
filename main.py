from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns

dados = pd.read_csv('large_data_convertido.csv')
dados.dropna(inplace=True)

x = np.array(dados.iloc[:, :-1])
y = np.array(dados['TYPE'])

#divisão base de treino e teste - 30% teste e 70% treino
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, train_size=0.7)

neighboor = [3, 5, 7]

previsoes = []
for n in neighboor:
    knn = KNeighborsClassifier(n)
    knn.fit(x_train, y_train)
    previsto = knn.predict(x_test)
    previsoes.append(previsto)
    acuracia = accuracy_score(y_test, previsto) * 100
    matriz_confusao = confusion_matrix(y_test, previsto)
    f1 = f1_score(y_test, previsto, average='weighted')
    precisao = precision_score(y_test, previsto, average='weighted')
    recall = recall_score(y_test, previsto, average='weighted')
    
    print(f"\n--- Resultados para k={n} ---")
    print(f"Acurácia: {acuracia:.2f}%")
    print(f"Precisão: {precisao:.2f}")
    print(f"Recall (Sensibilidade): {recall:.2f}")
    print(f"F1-Score: {f1:.2f}")
    
    # imagem da matriz de confusão
    plt.figure(figsize=(5, 4))
    sns.heatmap(matriz_confusao, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Matiz de Confusão para k = {n}')
    plt.xlabel('Predito'); plt.ylabel('Verdadeiro')
    plt.savefig(f'matriz_confusao_k_{n}.png', bbox_inches='tight')
    plt.close()
    
    