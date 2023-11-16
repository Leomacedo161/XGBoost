import os
import csv
import random

def gerar_questoes(codigo_operador, num_questoes=400000):
    questoes = set()  # Usando um conjunto para garantir unicidade dentro do operador
    while len(questoes) < num_questoes:
        valor1 = random.randint(1, 999)
        valor2 = random.randint(1, 999)
        resultado = calcular_resultado(valor1, valor2, codigo_operador)
        questao = (valor1, codigo_operador, valor2, resultado)  # Corrigido a ordem dos elementos
        questoes.add(questao)  # Adiciona diretamente ao conjunto
    return questoes  # Retorna o conjunto diretamente

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
        # Garantir resultado inteiro para a divisão
        return valor1 // valor2

def salvar_em_csv(questoes, codigo_operador, pasta='dataset'):
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
    print(f"{nome_arquivo} criado com sucesso.")

def gerar_questoes_mistas(num_questoes_por_operador=100000):
    questoes_mistas = set()
    for codigo_operador in range(4):
        questoes = gerar_questoes(codigo_operador, num_questoes_por_operador)
        questoes_mistas.update(questoes)
    return questoes_mistas

def main():
    # Defina a quantidade desejada de questões por operador
    num_questoes_por_operador = 100000

    # Criar datasets separados por operador
    for codigo_operador in range(4):
        questoes = gerar_questoes(codigo_operador, num_questoes_por_operador)
        salvar_em_csv(questoes, codigo_operador)

    # Criar dataset com questões mistas
    pasta_mista = 'dataset2'
    if not os.path.exists(pasta_mista):
        os.makedirs(pasta_mista)

    questoes_mistas = gerar_questoes_mistas(num_questoes_por_operador)
    salvar_em_csv(questoes_mistas, 0, pasta_mista)  # 0 é apenas um código fictício para questões mistas

if __name__ == "__main__":
    main()
