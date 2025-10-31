import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob

# Путь к файлам CSV
file_paths = glob.glob("/Users/kirillkiselev/Desktop/Диплом/Database/*.csv")  # Все CSV в папке

crypto_data = {}

# Загрузка данных
for file in file_paths:
    crypto_name = file.split('/')[-1].split('.')[0]  # Имя актива из названия файла
    df = pd.read_csv(file, index_col=0, parse_dates=True)

    # Проверяем наличие столбца "Цена" и конвертируем в числовой формат
    if 'Цена' in df.columns:
        df['Цена'] = df['Цена'].astype(str).str.replace('.', '').str.replace(',', '.').astype(float)
        crypto_data[crypto_name] = df['Цена']
    else:
        raise ValueError(f"Файл {file} не содержит столбца 'Цена'")

# Объединение всех данных в один DataFrame
data = pd.DataFrame(crypto_data)

# Расчет дневной доходности
returns = data.pct_change().dropna()

# Средняя годовая доходность и ковариационная матрица
mean_returns = returns.mean() * 252  # Годовая доходность
cov_matrix = returns.cov() * 252  # Годовая ковариационная матрица

# Функция для вычисления Sortino Ratio
def sortino_ratio(returns, risk_free_rate=0):
    downside_returns = returns[returns < 0]  # Оставляем только отрицательные доходности
    downside_std = downside_returns.std() * np.sqrt(252)  # Годовое отклонение вниз
    return (returns.mean() * 252 - risk_free_rate) / downside_std if downside_std != 0 else np.nan

# Функция для вычисления Value at Risk (VaR)
def value_at_risk(returns, confidence_level=0.05):
    return np.percentile(returns, 100 * confidence_level)

# Количество симуляций
num_portfolios = 10000

# Массивы для хранения результатов
asset_names = list(data.columns)
#results = np.zeros((3 + len(asset_names), num_portfolios))
results = np.zeros((5 + len(asset_names), num_portfolios))
asset_names = list(data.columns)

for i in range(num_portfolios):
    weights = np.random.random(len(asset_names))
    weights /= np.sum(weights)  # Нормализация весов

    # Рассчет доходности и риска
    portfolio_return = np.sum(weights * mean_returns)
    portfolio_stddev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe_ratio = portfolio_return / portfolio_stddev  # Коэффициент Шарпа
    sortino = sortino_ratio(np.dot(returns, weights))
    var_95 = value_at_risk(np.dot(returns, weights), 0.05)

    # Сохранение результатов
    results[0, i] = portfolio_return
    results[1, i] = portfolio_stddev
    results[2, i] = sharpe_ratio
    results[3, i] = sortino
    results[4, i] = var_95
    #results[3:3+len(asset_names), i] = weights
    results[5:5 + len(asset_names), i] = weights

# Находим портфель с максимальным коэффициентом Шарпа
max_sharpe_idx = np.argmax(results[2])
max_sharpe_portfolio = results[:, max_sharpe_idx]

# Визуализация портфелей
plt.figure(figsize=(10, 6))
plt.scatter(results[1, :], results[0, :], c=results[2, :], cmap='viridis', marker='o')
plt.colorbar(label='Sharpe Ratio')
plt.xlabel('Риск (Std Dev)')
plt.ylabel('Доходность')
plt.title('Оптимизация портфеля методом Монте-Карло')

# Выделение лучшего портфеля
plt.scatter(max_sharpe_portfolio[1], max_sharpe_portfolio[0], color='red', marker='*', s=200, label='Макс. Sharpe')
plt.legend()
plt.show()

# Вывод оптимального распределения активов
#optimal_weights = {asset: weight for asset, weight in zip(asset_names, max_sharpe_portfolio[3:3+len(asset_names)])}
optimal_weights = {asset: weight for asset, weight in zip(asset_names, max_sharpe_portfolio[5:5+len(asset_names)])}

weights_df = pd.DataFrame(list(optimal_weights.items()), columns=['Asset', 'Weight'])
weights_df.to_csv('/Users/kirillkiselev/Desktop/Диплом/optimal_portfolio_MCS.csv', index=False)
print("Оптимальное распределение активов сохранено в 'optimal_portfolio_MCS.csv'")

# Вывод метрик для оптимального портфеля
metrics_df = pd.DataFrame({
    'Sharpe Ratio': [max_sharpe_portfolio[2]],
    'Sortino Ratio': [max_sharpe_portfolio[3]],
    'VaR 95%': [max_sharpe_portfolio[4]]
})
metrics_df.to_csv('/Users/kirillkiselev/Desktop/Диплом/МНСК конференция/optimal_portfolio_metrics.csv', index=False)

