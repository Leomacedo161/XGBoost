import os
import csv
import random

def gerar_questoes(codigo_operador, num_questoes=100000):
    questoes = []
    for _ in range(num_questoes):
        valor1 = random.randint(1, 999)
        valor2 = random.randint(1, 999)
        resultado = calcular_resultado(valor1, valor2, codigo_operador)
        questoes.append([valor1, codigo_operador, valor2, resultado])
    return questoes

def calcular_resultado(valor1, valor2, codigo_operador):
    if codigo_operador == 0:
        return valor1 + valor2
    elif codigo_operador == 1:
        return valor1 - valor2
    elif codigo_operador == 2:
        return valor1 * valor2
    elif codigo_operador == 3:
        # Evitar divisão por zero
        if valor2 == 0:
            valor2 = 1
        return valor1 / valor2

def salvar_em_csv(questoes, codigo_operador):
    nomes_operadores = ['soma', 'subtracao', 'multiplicacao', 'divisao']
    nome_operador = nomes_operadores[codigo_operador]
    nome_pasta = 'dataset'
    if not os.path.exists(nome_pasta):
        os.makedirs(nome_pasta)
    nome_arquivo = os.path.join(nome_pasta, f"{nome_operador}.csv")
    with open(nome_arquivo, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Valor1', 'Codigo_Operador', 'Valor2', 'Resultado'])
        csvwriter.writerows(questoes)
    print(f"{nome_arquivo} criado com sucesso.")

for codigo_operador in range(4):
    questoes = gerar_questoes(codigo_operador)
    salvar_em_csv(questoes, codigo_operador)
