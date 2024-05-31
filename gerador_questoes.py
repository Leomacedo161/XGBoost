import os
import csv
import random
import logging
from multiprocessing import Pool

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def gerar_questoes(codigo_operador, num_questoes=40000, seed=None):
    if seed is not None:
        random.seed(seed)  # Definir semente para reprodutibilidade
    questoes = set()
    while len(questoes) < num_questoes:
        valor1 = random.randint(1, 100)
        valor2 = random.randint(1, 100)
        resultado = calcular_resultado(valor1, valor2, codigo_operador)
        questao = (valor1, codigo_operador, valor2, resultado)
        questoes.add(questao)
    return questoes

def calcular_resultado(valor1, valor2, codigo_operador):
    if codigo_operador == 0:
        return valor1 + valor2
    elif codigo_operador == 1:
        return valor1 - valor2
    elif codigo_operador == 2:
        return valor1 * valor2
    elif codigo_operador == 3:
        if valor2 == 0:
            valor2 = 1
        return round(valor1 / valor2, 3)

def salvar_em_csv(questoes, codigo_operador, pasta='datasetLista'):
    nomes_operadores = ['soma', 'subtracao', 'multiplicacao', 'divisao']
    nome_operador = nomes_operadores[codigo_operador]
    nome_pasta = pasta
    if not os.path.exists(nome_pasta):
        os.makedirs(nome_pasta)
    nome_arquivo = os.path.join(nome_pasta, f"{nome_operador}.csv")
    with open(nome_arquivo, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Valor1', 'Codigo_Operador', 'Valor2', 'Resultado'])
        csvwriter.writerows(questoes)
    logging.info(f"{nome_arquivo} criado com sucesso.")

def gerar_questoes_mistas(num_questoes_por_operador=10000, seed=None):
    questoes_mistas = set()
    for codigo_operador in range(4):
        questoes = gerar_questoes(codigo_operador, num_questoes_por_operador, seed)
        questoes_mistas.update(questoes)
    return questoes_mistas

def processar_operador(args):
    codigo_operador, num_questoes_por_operador, seed = args
    questoes = gerar_questoes(codigo_operador, num_questoes_por_operador, seed)
    salvar_em_csv(questoes, codigo_operador)

def main():
    num_questoes_por_operador = 10000
    seed = 42  # Define a semente para reprodutibilidade

    # Criar datasets separados por operador
    args = [(codigo_operador, num_questoes_por_operador, seed) for codigo_operador in range(4)]
    with Pool(processes=4) as pool:
        pool.map(processar_operador, args)

    # Criar dataset com questões mistas
    pasta_mista = 'datasetUnico'
    if not os.path.exists(pasta_mista):
        os.makedirs(pasta_mista)

    questoes_mistas = gerar_questoes_mistas(num_questoes_por_operador, seed)
    salvar_em_csv(questoes_mistas, 0, pasta_mista)  # 0 é apenas um código fictício para questões mistas

if __name__ == "__main__":
    main()
