import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Exemplo de dados fictícios de clientes
data = {
    'ClienteID': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Idade': [25, 34, 45, 31, 35, 40, 52, 23, 43, 50],
    'Renda_Anual': [50000, 60000, 80000, 45000, 70000, 100000, 120000, 30000, 85000, 110000],
    'Pontuacao_Gastos': [60, 70, 50, 80, 65, 85, 40, 75, 55, 90]
}

# Convertendo os dados para um DataFrame
df = pd.DataFrame(data)

# Selecionando as características para clustering
X = df[['Idade', 'Renda_Anual', 'Pontuacao_Gastos']]

# Padronizando os dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Aplicando K-means com 3 clusters (segmentos de mercado)
kmeans = KMeans(n_clusters=1, random_state=0)
df['Segmento'] = kmeans.fit_predict(X_scaled)

# Visualizando os resultados
print(df)

# Plotando os clusters
plt.figure(figsize=(8, 6))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=df['Segmento'], cmap='viridis')
plt.xlabel('Idade (Padronizada)')
plt.ylabel('Renda Anual (Padronizada)')
plt.title('Segmentação de Mercado com K-Means')
plt.show()
