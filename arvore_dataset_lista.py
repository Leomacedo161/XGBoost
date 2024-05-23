import os
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Carregar os dados
def carregar_dados(caminho_arquivo):
    return pd.read_csv(caminho_arquivo)

# Treinar o modelo
def treinar_modelo(dados, param_grid, model_type='decision_tree'):
    X = dados[['Valor1', 'Codigo_Operador', 'Valor2']]
    y = dados['Resultado']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalização dos dados
    scaler = StandardScaler()
    
    if model_type == 'decision_tree':
        from sklearn.tree import DecisionTreeRegressor
        model = DecisionTreeRegressor(random_state=42)
    elif model_type == 'random_forest':
        model = RandomForestRegressor(random_state=42)
    elif model_type == 'gradient_boosting':
        model = GradientBoostingRegressor(random_state=42)
    
    pipeline = Pipeline([('scaler', scaler), ('model', model)])
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error', error_score='raise')
    grid_search.fit(X_train, y_train)

    # Melhores hiperparâmetros
    print(f"\nMelhores Hiperparâmetros para {model_type.replace('_', ' ').title()}: {grid_search.best_params_}")

    # Treinamento do modelo com os melhores hiperparâmetros
    modelo = grid_search.best_estimator_

    # Avaliação do modelo nos dados de teste
    y_pred_test = modelo.predict(X_test)

    # Avaliação de métricas nos dados de teste
    mae_test = mean_absolute_error(y_test, y_pred_test)
    mse_test = mean_squared_error(y_test, y_pred_test)
    r2_test = r2_score(y_test, y_pred_test)

    print(f"\nErro Absoluto Médio (MAE) nos dados de teste: {mae_test:.2f}")
    print(f"Erro Quadrático Médio (MSE) nos dados de teste: {mse_test:.2f}")
    print(f"Coeficiente de Determinação (R^2) nos dados de teste: {r2_test:.2f}")

    return modelo, X_test, y_test

# Prever manualmente
def prever_manualmente(modelo):
    while True:
        print("\nPrever Manualmente:")
        valor1 = float(input("Digite o Valor1: "))
        codigo_operador = int(input("Digite o Codigo_Operador: "))
        valor2 = float(input("Digite o Valor2: "))

        entrada = [[valor1, codigo_operador, valor2]]
        previsao = modelo.predict(entrada)[0]
        print(f"\nA previsão do modelo é: {previsao}")

        continuar = input("Deseja continuar? (Digite 's' para sair): ").lower()
        if continuar == 's':
            break

# Exibir resultados
def exibir_resultados(modelo, X_test, y_test):
    y_pred = modelo.predict(X_test)

    # Avaliação de métricas
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Contagem de valores corretos e incorretos
    valores_corretos = sum((y_test - y_pred) ** 2 == 0)
    valores_incorretos = len(y_test) - valores_corretos

    # Porcentagem de acertos
    porcentagem_acertos = (valores_corretos / len(y_test)) * 100

    print(f"\nErro Absoluto Médio (MAE): {mae:.2f}")
    print(f"Erro Quadrático Médio (MSE): {mse:.2f}")
    print(f"Coeficiente de Determinação (R^2): {r2:.2f}")
    print(f"Valores Corretos: {valores_corretos}")
    print(f"Valores Incorretos: {valores_incorretos}")
    print(f"Porcentagem de Acertos: {porcentagem_acertos:.2f}%")

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
    nome_pasta = 'datasetLista'
    dados_completos = pd.DataFrame()

    for nome_arquivo in os.listdir(nome_pasta):
        if nome_arquivo.endswith('.csv'):
            caminho_arquivo = os.path.join(nome_pasta, nome_arquivo)
            dados = carregar_dados(caminho_arquivo)
            dados_completos = pd.concat([dados_completos, dados], ignore_index=True)

    # Definir grid de hiperparâmetros
    param_grid = {
        'model__max_depth': [None, 5, 10],
        'model__min_samples_split': [2, 5],
        'model__min_samples_leaf': [1, 2],
        'model__max_features': ['sqrt', 'log2'],
        'model__max_leaf_nodes': [None, 5, 10],
        'model__ccp_alpha': [0.0, 0.01]
    }

    # Treinar e avaliar modelos
    for model_type in ['decision_tree', 'random_forest', 'gradient_boosting']:
        print(f"\nTreinando modelo: {model_type.replace('_', ' ').title()}")
        modelo, X_test, y_test = treinar_modelo(dados_completos.sample(frac=0.8, random_state=42), param_grid, model_type)
        print(f"\nResultados para {model_type.replace('_', ' ').title()}:")
        exibir_resultados(modelo, X_test, y_test)

        # Prever manualmente após o treinamento
        prever_manualmente(modelo)

if __name__ == "__main__":
    executar_analise()
