import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Definindo parâmetros do dataset fictício
np.random.seed(42)
dates = pd.date_range(start='2022-01-01', end='2023-12-31')
sales = 200 + (np.sin(np.linspace(0, 3.14 * 2, len(dates))) * 50) + np.random.normal(0, 20, len(dates))

# Criando DataFrame
data = pd.DataFrame({'date': dates, 'sales': sales})
data['sales'] = data['sales'].astype(int)
data.to_csv('historical_sales_data.csv', index=False)

# Visualizar os dados gerados
plt.figure(figsize=(10, 5))
plt.plot(data['date'], data['sales'])
plt.title('Vendas Diárias Fictícias')
plt.xlabel('Data')
plt.ylabel('Vendas')
plt.show()
