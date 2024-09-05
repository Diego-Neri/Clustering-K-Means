# 1. Importar as bibliotecas necessárias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# 2. Criar um dataset fictício de clientes (renda anual e gasto em produtos)
data = {
    'Renda_Anual': [15, 16, 17, 18, 19, 25, 26, 28, 30, 35, 40, 60, 65, 70, 80],
    'Gasto_Anual': [39, 81, 6, 77, 40, 76, 20, 65, 55, 50, 45, 88, 98, 70, 100]
}

# Convertendo o dicionário em um DataFrame do Pandas
df = pd.DataFrame(data)

# 3. Visualizar os dados
plt.scatter(df['Renda_Anual'], df['Gasto_Anual'], color='blue')
plt.xlabel('Renda Anual (milhares)')
plt.ylabel('Gasto Anual (milhares)')
plt.title('Distribuição de Clientes')
plt.show()

# 4. Aplicar o algoritmo K-means
# Inicializamos o K-means com o número de clusters que desejamos (neste caso, 3)
kmeans = KMeans(n_clusters=3)

# Ajustamos o modelo aos dados de renda e gasto
kmeans.fit(df)

# 5. Obter os clusters (labels) para cada ponto de dados
df['Cluster'] = kmeans.labels_

# 6. Visualizar os clusters formados
colors = ['red', 'green', 'blue']  # Cores para diferenciar os clusters

for i in range(3):  # Para cada cluster, plotamos os pontos pertencentes a ele
    cluster_points = df[df['Cluster'] == i]
    plt.scatter(cluster_points['Renda_Anual'], cluster_points['Gasto_Anual'], color=colors[i], label=f'Cluster {i+1}')

plt.xlabel('Renda Anual (milhares)')
plt.ylabel('Gasto Anual (milhares)')
plt.title('Segmentação de Clientes com K-means')
plt.legend()
plt.show()
