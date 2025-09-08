import numpy as np
import random
from typing import Tuple, List, Dict

def normalizar_dados(dados: np.ndarray) -> np.ndarray:
    """
    Função da normalização dos dados entre 0 e 1
    """
    dados_norm = dados.copy()
    
    # Normalizar IMC (coluna 1): de 16 a 40
    dados_norm[:, 1] = (dados[:, 1] - 16) / (40 - 16)
    
    # Normalizar Glicemia (coluna 2): de 70 a 126
    dados_norm[:, 2] = (dados[:, 2] - 70) / (126 - 70)
    
    # Bias (coluna 0) e saídas (colunas 3 e 4) permanecem iguais
    
    return dados_norm

def sigmoid(x: float) -> float:
    """
    Função sigmoid para ativação
    """
    return 1 / (1 + np.exp(-x))

def calcular_metricas(tp: int, tn: int, fp: int, fn: int) -> Dict[str, float]:
    """
    Calcula as métricas de avaliação
    """
    total = tp + tn + fp + fn
    
    # Evitar divisão por zero
    acuracia = (tp + tn) / total if total > 0 else 0
    precisao = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = (2 * precisao * recall) / (precisao + recall) if (precisao + recall) > 0 else 0
    
    return {
        'acuracia': acuracia * 100,
        'precisao': precisao * 100,
        'recall': recall * 100,
        'f1_score': f1_score * 100
    }

def main():
    # Parâmetros da rede
    epocas = 1000
    taxa_aprendizado = 0.1
    
    # Dados de entrada
    # [bias, imc, glicemia, saida_1(com diabetes), saida_2(sem diabetes)]
    dados = np.array([
        [1, 16, 70, 0, 1],   # Sem diabetes
        [1, 18, 80, 0, 1],   # Sem diabetes
        [1, 20, 90, 0, 1],   # Sem diabetes
        [1, 40, 126, 1, 0],  # Com diabetes
        [1, 30, 125, 1, 0],  # Com diabetes
        [1, 22, 100, 0, 1],  # Sem diabetes
        [1, 35, 122, 1, 0],  # Com diabetes
        [1, 16, 75, 0, 1],   # Sem diabetes
        [1, 25, 110, 0, 1],  # Sem diabetes
        [1, 38, 126, 1, 0],  # Com diabetes
        [1, 25, 70, 0, 1],   # Sem diabetes
        [1, 20, 72, 0, 1],   # Sem diabetes
        [1, 35, 99, 1, 0],   # Com diabetes
        [1, 32, 120, 0, 1],  # Sem diabetes
        [1, 31, 110, 0, 1],  # Sem diabetes
        [1, 33, 95, 0, 1]    # Sem diabetes
    ], dtype=float)
    
    # Normalização dos dados
    dados_norm = normalizar_dados(dados)
    
    # Inicialização dos pesos da rede
    # Camada Oculta: 3 neurônios com 3 entradas cada (bias, IMC, glicemia)
    pesos_camada_oculta = np.random.uniform(-1, 1, (3, 3))
    
    # Camada de Saída: 2 neurônios com 4 entradas cada (bias + 3 neurônios ocultos)
    pesos_camada_saida = np.random.uniform(-1, 1, (2, 4))
    
    # Separação dos dados
    indices_treino = [0, 1, 2, 3, 4, 5, 6, 8, 13, 15]
    indices_teste = [10, 11, 12, 7, 14, 9]
    
    # Treinamento da rede neural
    print("Iniciando treinamento da rede neural...")
    
    for epoca in range(epocas):
        # Aleatorizar ordem dos dados de treino
        random.shuffle(indices_treino)
        
        for idx in indices_treino:
            linha = dados_norm[idx]
            
            # ===== PROPAGAÇÃO PARA FRENTE =====
            
            # Camada Oculta (3 neurônios)
            camada_oculta = np.zeros(3)
            for i in range(3):
                soma = 0
                for j in range(3):  # bias, IMC, glicemia
                    soma += linha[j] * pesos_camada_oculta[i][j]
                camada_oculta[i] = sigmoid(soma)
            
            # Camada de Saída (2 neurônios)
            camada_saida = np.zeros(2)
            for i in range(2):
                camada_saida[i] = pesos_camada_saida[i][0]  # Bias
                # Conexões dos neurônios ocultos
                for j in range(1, 4):
                    camada_saida[i] += camada_oculta[j-1] * pesos_camada_saida[i][j]
            
            # ===== RETROPROPAGAÇÃO =====
            
            # Erro da camada de saída
            delta_saida = np.zeros(2)
            for i in range(2):
                erro = linha[i+3] - camada_saida[i]  # Saída esperada - saída atual
                delta_saida[i] = erro * 1  # Derivada linear (função identidade)
            
            # Erro da camada oculta
            delta_oculta = np.zeros(3)
            for i in range(3):
                soma = 0
                for j in range(2):
                    soma += delta_saida[j] * pesos_camada_saida[j][i+1]
                # Derivada da sigmoid: f'(x) = f(x) * (1 - f(x))
                delta_oculta[i] = camada_oculta[i] * (1 - camada_oculta[i]) * soma
            
            # ===== ATUALIZAÇÃO DOS PESOS =====
            
            # Atualizar pesos - Camada de saída
            for i in range(2):
                pesos_camada_saida[i][0] += taxa_aprendizado * delta_saida[i] * 1  # Bias
                for j in range(1, 4):
                    pesos_camada_saida[i][j] += taxa_aprendizado * delta_saida[i] * camada_oculta[j-1]
            
            # Atualizar pesos - Camada oculta
            for i in range(3):
                pesos_camada_oculta[i][0] += taxa_aprendizado * delta_oculta[i] * 1  # Bias
                for j in range(1, 3):
                    pesos_camada_oculta[i][j] += taxa_aprendizado * delta_oculta[i] * linha[j]
        
        # Exibir progresso
        if epoca % 100 == 0:
            print(f"Época: {epoca} concluída")
    
    print("Treinamento concluído!\n")
    
    # ===== TESTE DA REDE =====
    
    def testar_rede_com_metricas(indices: List[int], nome_conjunto: str) -> Dict:
        """Testa a rede com um conjunto de dados e calcula métricas"""
        print(f"=== RESULTADOS COM DADOS DE {nome_conjunto.upper()} ===")
        
        # Inicializar contadores da matriz de confusão
        tp = tn = fp = fn = 0
        
        for idx in indices:
            linha = dados_norm[idx]
            
            # Propagação para teste
            camada_oculta = np.zeros(3)
            for i in range(3):
                soma = 0
                for j in range(3):
                    soma += linha[j] * pesos_camada_oculta[i][j]
                camada_oculta[i] = sigmoid(soma)
            
            camada_saida = np.zeros(2)
            for i in range(2):
                camada_saida[i] = pesos_camada_saida[i][0]
                for j in range(1, 4):
                    camada_saida[i] += camada_oculta[j-1] * pesos_camada_saida[i][j]
            
            # Determinar classes
            previsto_diabetes = camada_saida[0] > camada_saida[1]  # True se previu diabetes
            real_diabetes = dados[idx][3] == 1  # True se realmente tem diabetes
            
            classe_prevista = "COM DIABETES" if previsto_diabetes else "SEM DIABETES"
            classe_real = "COM DIABETES" if real_diabetes else "SEM DIABETES"
            
            # Calcular matriz de confusão
            # Positivo = COM DIABETES, Negativo = SEM DIABETES
            if previsto_diabetes and real_diabetes:
                tp += 1
                resultado = "(TP)"
            elif not previsto_diabetes and not real_diabetes:
                tn += 1
                resultado = "(TN)"
            elif previsto_diabetes and not real_diabetes:
                fp += 1
                resultado = "(FP)"
            else:  # not previsto_diabetes and real_diabetes
                fn += 1
                resultado = "(FN)"
            
            print(f"Paciente  {idx+1}:\tIMC={dados[idx][1]:.0f},\t"
                  f"Glic={dados[idx][2]:.0f} \t->\t{classe_prevista} "
                  f"[{camada_saida[0]:.0f}, {camada_saida[1]:.0f}]\t{resultado}")
        
        total = len(indices)
        acertos = tp + tn
        print(f"Acertos no {nome_conjunto}: {acertos}/{total}")
        
        # Calcular métricas
        metricas = calcular_metricas(tp, tn, fp, fn)
        
        return {
            'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
            'acertos': acertos, 'total': total,
            'metricas': metricas
        }
    
    # Teste com dados de treino
    resultado_treino = testar_rede_com_metricas(indices_treino, "treino")
    print()
    
    # Teste com dados de teste
    resultado_teste = testar_rede_com_metricas(indices_teste, "teste")
    print()
    
    # ===== DESEMPENHO DA REDE COM MÉTRICAS COMPLETAS =====
    print("=== DESEMPENHO DA REDE ===")
    print(f"Acurácia no treino: {resultado_treino['metricas']['acuracia']:.2f}%")
    print(f"Acurácia no teste: {resultado_teste['metricas']['acuracia']:.2f}%")
    print()
    print("=== MÉTRICAS DETALHADAS - DADOS DE TESTE ===")
    print(f"Acurácia = (TP + TN) / (TP + TN + FP + FN) = {resultado_teste['metricas']['acuracia']:.2f}%")
    print(f"Precisão = TP / (TP + FP) = {resultado_teste['metricas']['precisao']:.2f}%")
    print(f"Recall = TP / (TP + FN) = {resultado_teste['metricas']['recall']:.2f}%")
    print(f"F1-score = 2 * (Precisão * Recall) / (Precisão + Recall) = {resultado_teste['metricas']['f1_score']:.2f}%")
    print()
    
    # Avaliação geral
    accuracy_teste = resultado_teste['metricas']['acuracia']
    if accuracy_teste > 80:
        print("Rede generalizando bem!")
    elif accuracy_teste > 60:
        print("Rede com desempenho moderado.")
    else:
        print("Rede com overfitting (apenas memorizou os dados de treino).")
    
    # ===== ANÁLISE DOS CRITÉRIOS CLÍNICOS =====
    print("\n=== CRITÉRIOS CLÍNICOS UTILIZADOS ===")
    print("IMC (Índice de Massa Corporal):")
    print("  • Abaixo de 18,5: baixo peso")
    print("  • Entre 18,5 e 24,9: peso normal")
    print("  • De 25 a 29,9: sobrepeso")
    print("  • 30 ou mais: obesidade")
    print("\nGlicemia em jejum:")
    print("  • ≤ 100 mg/dL: normal")
    print("  • Entre 100 e 125 mg/dL: pré-diabetes")
    print("  • ≥ 126 mg/dL: diabetes")

if __name__ == "__main__":
    # Seed para reprodutibilidade
    np.random.seed(42)
    random.seed(42)
    
    main()