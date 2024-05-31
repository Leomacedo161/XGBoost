import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

# Carregar os dados gerados
def carregar_dados(nome_arquivo):
    return pd.read_csv(nome_arquivo)

# Criar novas features
def criar_features(data):
    data['Soma'] = data['Valor1'] + data['Valor2']
    data['Diferenca'] = data['Valor1'] - data['Valor2']
    data['Produto'] = data['Valor1'] * data['Valor2']
    data['Razao'] = data['Valor1'] / (data['Valor2'] + 1e-5)  # Adicionar um pequeno valor para evitar divisão por zero
    return data

# Treinar o modelo de XGBoost com ajuste de hiperparâmetros
def treinar_modelo(X, y):
    param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'min_child_weight': [1, 3, 5],
        'subsample': [0.6, 0.8, 1.0]
    }
    xgb = XGBRegressor(random_state=42)
    grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X, y)
    return grid_search.best_estimator_, grid_search.best_params_['learning_rate']

# Avaliar o modelo
def avaliar_modelo(model, X_test, y_test):
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    return r2, mae, mse, y_pred

# Plotar os resultados
def plotar_resultados(y_test, y_pred):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, edgecolors=(0, 0, 0))
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    plt.xlabel('Valores Reais')
    plt.ylabel('Valores Preditos')
    plt.title('Valores Reais vs. Valores Preditos')
    plt.show()

# Calcular o índice de acertos com tolerância ajustada
def calcular_indice_acertos(y_test, y_pred, operacao):
    if operacao == 'soma' or operacao == 'subtracao':
        tolerancia = 2
    elif operacao == 'multiplicacao':
        tolerancia = 5
    elif operacao == 'divisao':
        tolerancia = 3
    else:
        tolerancia = 0
    
    acertos = np.sum(np.isclose(y_test, y_pred, atol=tolerancia))
    erros = len(y_test) - acertos
    indice_acertos = acertos / len(y_test)
    return indice_acertos, acertos, erros

# Plotar gráfico de acertos e erros
def plotar_acertos_erros(acertos_total, erros_total):
    labels = ['Acertos', 'Erros']
    valores = [acertos_total, erros_total]
    
    plt.figure(figsize=(10, 6))
    plt.bar(labels, valores, color=['blue', 'red'])
    plt.xlabel('Categoria')
    plt.ylabel('Quantidade')
    plt.title('Total de Acertos e Erros')
    plt.show()

# Main
def main_ml():
    # Nome do arquivo a ser processado
    nome_arquivo = 'datasetUnico/soma.csv'
    
    # Carregar os dados
    data = carregar_dados(nome_arquivo)
    
    # Criar novas features
    data = criar_features(data)
    
    # Dividir os dados em features (X) e target (y)
    X = data[['Valor1', 'Valor2', 'Soma', 'Diferenca', 'Produto', 'Razao']]
    y = data['Resultado']
    
    # Normalizar os dados
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Dividir os dados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # Treinar o modelo
    model, learning_rate = treinar_modelo(X_train, y_train)
    
    # Avaliar o modelo
    r2, mae, mse, y_pred = avaliar_modelo(model, X_test, y_test)
    
    # Definir uma tolerância única ou ajustar conforme necessário
    operacao = 'mixed'  # Operação mista
    indice_acertos, acertos, erros = calcular_indice_acertos(y_test, y_pred, operacao)
    
    # Imprimir as métricas de avaliação
    print(f"R^2: {r2}")
    print(f"MAE: {mae}")
    print(f"MSE: {mse}")
    print(f"Índice de Acertos: {indice_acertos}")
    print(f"Learning Rate: {learning_rate}")
    
    # Plotar resultados reais vs preditos
    plotar_resultados(y_test, y_pred)
    
    # Plotar gráfico de acertos e erros
    plotar_acertos_erros(acertos, erros)

if __name__ == "__main__":
    main_ml()
