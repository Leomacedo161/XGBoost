import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

class ArvoreDatasetLista:
    def __init__(self):
        pass

    def carregar_dados(self, nome_arquivo):
        return pd.read_csv(nome_arquivo)

    def treinar_modelo(self, X, y):
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

    def avaliar_modelo(self, model, X_test, y_test):
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        return r2, mae, mse, y_pred

    def calcular_indice_acertos(self, y_test, y_pred, operacao):
        if operacao == 0 or operacao == 1:
            tolerancia = 2
        elif operacao == 2:
            tolerancia = 5
        elif operacao == 3:
            tolerancia = 3
        else:
            tolerancia = 0

        acertos = np.sum(np.isclose(y_test, y_pred, atol=tolerancia))
        erros = len(y_test) - acertos
        indice_acertos = acertos / len(y_test)
        return indice_acertos, acertos, erros

    def plotar_acertos_erros(self, acertos_total, erros_total):
        labels = ['Acertos', 'Erros']
        valores = [acertos_total, erros_total]

        plt.figure(figsize=(10, 6))
        plt.bar(labels, valores, color=['blue', 'red'])
        plt.xlabel('Categoria')
        plt.ylabel('Quantidade')
        plt.title('Total de Acertos e Erros')
        plt.show()

    def main_ml(self):
        arquivos_operacoes = [
            ('datasetLista/soma.csv', 0),
            ('datasetLista/subtracao.csv', 1),
            ('datasetLista/multiplicacao.csv', 2),
            ('datasetLista/divisao.csv', 3)
        ]

        r2_list = []
        mae_list = []
        mse_list = []
        indice_acertos_list = []
        learning_rates = []

        acertos_total = 0
        erros_total = 0

        for nome_arquivo, operacao in arquivos_operacoes:
            data = self.carregar_dados(nome_arquivo)
            X = data[['Valor1', 'Codigo_Operador', 'Valor2']]
            y = data['Resultado']
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
            model, learning_rate = self.treinar_modelo(X_train, y_train)
            r2, mae, mse, y_pred = self.avaliar_modelo(model, X_test, y_test)
            indice_acertos, acertos, erros = self.calcular_indice_acertos(y_test, y_pred, operacao)
            r2_list.append(r2)
            mae_list.append(mae)
            mse_list.append(mse)
            indice_acertos_list.append(indice_acertos)
            learning_rates.append(learning_rate)
            acertos_total += acertos
            erros_total += erros

        media_r2 = np.mean(r2_list)
        media_mae = np.mean(mae_list)
        media_mse = np.mean(mse_list)
        media_indice_acertos = np.mean(indice_acertos_list)
        media_learning_rate = np.mean(learning_rates)

        print(f"Média R^2: {media_r2}")
        print(f"Média MAE: {media_mae}")
        print(f"Média MSE: {media_mse}")
        print(f"Média Índice de Acertos: {media_indice_acertos}")
        print(f"Média Learning Rate: {media_learning_rate}")

        # self.plotar_acertos_erros(acertos_total, erros_total)

        return media_r2, media_mae, media_mse, media_indice_acertos, media_learning_rate

    def executar(self):
        return self.main_ml()
