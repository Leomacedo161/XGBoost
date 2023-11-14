import os
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Carregar os dados
def carregar_dados(caminho_arquivo):
    return pd.read_csv(caminho_arquivo)

# Treinar a árvore de decisão
def treinar_arvore_decisao(dados):
    X = dados[['Valor1', 'Codigo_Operador', 'Valor2']]
    y = dados['Resultado']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalização dos dados
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Ajuste de hiperparâmetros
    param_grid = {
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    tree_model = DecisionTreeRegressor(random_state=42)
    grid_search = GridSearchCV(tree_model, param_grid, cv=3, scoring='neg_mean_squared_error')
    grid_search.fit(X_train_scaled, y_train)

    # Melhores hiperparâmetros
    print("\nMelhores Hiperparâmetros para Árvore de Decisão:", grid_search.best_params_)

    # Treinamento do modelo com os melhores hiperparâmetros
    modelo_arvore = grid_search.best_estimator_
    modelo_arvore.fit(X_train_scaled, y_train)

    # Avaliação do modelo nos dados de teste
    y_pred_test = modelo_arvore.predict(X_test_scaled)

    # Avaliação de métricas nos dados de teste
    mae_test = mean_absolute_error(y_test, y_pred_test)
    mse_test = mean_squared_error(y_test, y_pred_test)
    r2_test = r2_score(y_test, y_pred_test)

    print(f"\nErro Absoluto Médio (MAE) nos dados de teste: {mae_test:.2f}")
    print(f"Erro Quadrático Médio (MSE) nos dados de teste: {mse_test:.2f}")
    print(f"Coeficiente de Determinação (R^2) nos dados de teste: {r2_test:.2f}")

    return modelo_arvore, X_test_scaled, y_test

# Exibir resultados
def exibir_resultados(modelo, X_test, y_test):
    y_pred = modelo.predict(X_test)

    # Avaliação de métricas
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\nErro Absoluto Médio (MAE): {mae:.2f}")
    print(f"Erro Quadrático Médio (MSE): {mse:.2f}")
    print(f"Coeficiente de Determinação (R^2): {r2:.2f}")

    # Gráfico de dispersão com regressão linear
    plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='red', linewidth=2)
    plt.title('Valores Reais vs. Valores Previstos')
    plt.xlabel('Real')
    plt.ylabel('Previsto')
    plt.show()

    # Cálculo da variação natural dos dados
    amplitude_dados = y_test.max() - y_test.min()
    print(f"Amplitude (Variação Natural) dos Dados: {amplitude_dados:.2f}")

# Executar análise
def executar_analise():
    nome_pasta = 'dataset'
    dados_completos = pd.DataFrame()

    for nome_arquivo in os.listdir(nome_pasta):
        if nome_arquivo.endswith('.csv'):
            caminho_arquivo = os.path.join(nome_pasta, nome_arquivo)
            dados = carregar_dados(caminho_arquivo)
            dados_completos = pd.concat([dados_completos, dados], ignore_index=True)

    # Treinar Árvore de Decisão
    modelo_arvore, X_test_arvore, y_test_arvore = treinar_arvore_decisao(dados_completos)
    print("\nResultados para Árvore de Decisão:")
    exibir_resultados(modelo_arvore, X_test_arvore, y_test_arvore)

if __name__ == "__main__":
    executar_analise()
