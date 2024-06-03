import numpy as np
import os
import shutil
from arvore_dataset_unico import main_ml as main_ml_unico
from arvore_dataset_lista import ArvoreDatasetLista
from gerador_questoes import main as gerar_questoes

class Executor:
    def __init__(self):
        self.r2_scores_unico = []
        self.mae_scores_unico = []
        self.mse_scores_unico = []
        self.accuracy_scores_unico = []
        self.learning_rates_unico = []

        self.r2_scores_lista = []
        self.mae_scores_lista = []
        self.mse_scores_lista = []
        self.accuracy_scores_lista = []
        self.learning_rates_lista = []

    def main_ml_unico(self):
        # Chame a função main_ml_unico
        r2, mae, mse, accuracy, learning_rate = main_ml_unico()
        return r2, mae, mse, accuracy, learning_rate

    def main_ml_lista(self):
        # Chame a classe ArvoreDatasetLista e execute o método main_ml
        arvore_lista = ArvoreDatasetLista()
        r2, mae, mse, accuracy, learning_rate = arvore_lista.main_ml()
        return r2, mae, mse, accuracy, learning_rate

    def limpar_datasets(self):
        if os.path.exists('datasetLista'):
            shutil.rmtree('datasetLista')
        if os.path.exists('datasetUnico'):
            shutil.rmtree('datasetUnico')

    def executar(self):
        for _ in range(10):
            # Excluir e gerar novos datasets
            self.limpar_datasets()
            gerar_questoes()

            # Execute a função main_ml_unico e obtenha as métricas para 'unico'
            r2, mae, mse, accuracy, learning_rate = self.main_ml_unico()

            # Adicione as métricas às listas correspondentes
            self.r2_scores_unico.append(r2)
            self.mae_scores_unico.append(mae)
            self.mse_scores_unico.append(mse)
            self.accuracy_scores_unico.append(accuracy)
            self.learning_rates_unico.append(learning_rate)

            # Execute a função main_ml_lista e obtenha as métricas para 'lista'
            r2, mae, mse, accuracy, learning_rate = self.main_ml_lista()

            # Adicione as métricas às listas correspondentes
            self.r2_scores_lista.append(r2)
            self.mae_scores_lista.append(mae)
            self.mse_scores_lista.append(mse)
            self.accuracy_scores_lista.append(accuracy)
            self.learning_rates_lista.append(learning_rate)

        # Calcule a média das métricas para 'unico'
        avg_r2_unico = np.mean(self.r2_scores_unico)
        avg_mae_unico = np.mean(self.mae_scores_unico)
        avg_mse_unico = np.mean(self.mse_scores_unico)
        avg_accuracy_unico = np.mean(self.accuracy_scores_unico)
        avg_learning_rate_unico = np.mean(self.learning_rates_unico)

        # Calcule a média das métricas para 'lista'
        avg_r2_lista = np.mean(self.r2_scores_lista)
        avg_mae_lista = np.mean(self.mae_scores_lista)
        avg_mse_lista = np.mean(self.mse_scores_lista)
        avg_accuracy_lista = np.mean(self.accuracy_scores_lista)
        avg_learning_rate_lista = np.mean(self.learning_rates_lista)

        # Imprima as médias finais para 'unico'
        print(f"Average R^2 for 'unico': {avg_r2_unico}")
        print(f"Average MAE for 'unico': {avg_mae_unico}")
        print(f"Average MSE for 'unico': {avg_mse_unico}")
        print(f"Average Accuracy for 'unico': {avg_accuracy_unico}")
        print(f"Average Learning Rate for 'unico': {avg_learning_rate_unico}")

        # Imprima as médias finais para 'lista'
        print(f"Average R^2 for 'lista': {avg_r2_lista}")
        print(f"Average MAE for 'lista': {avg_mae_lista}")
        print(f"Average MSE for 'lista': {avg_mse_lista}")
        print(f"Average Accuracy for 'lista': {avg_accuracy_lista}")
        print(f"Average Learning Rate for 'lista': {avg_learning_rate_lista}")

if __name__ == '__main__':
    executor = Executor()
    executor.executar()
