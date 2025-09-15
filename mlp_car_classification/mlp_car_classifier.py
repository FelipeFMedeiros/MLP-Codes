import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict
from sklearn.model_selection import train_test_split


def carregar_e_preprocessar_dados(path_csv: str) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Lê o arquivo CSV, seleciona features e transforma o target em binário (Premium/Econômico).
    Retorna X (entradas com bias), y (saídas binárias) e DataFrame original.
    """
    print("=== CARREGANDO E PREPROCESSANDO DADOS ===")
    df = pd.read_csv(path_csv)
    print(f"Dataset carregado: {len(df)} carros")

    # Mostrar estatísticas básicas
    print(f"Ano: {df['year'].min()}-{df['year'].max()}")
    print(f"Preço: ${df['price'].min():,}-${df['price'].max():,}")
    print(f"Quilometragem: {df['mileage'].min():,}-{df['mileage'].max():,} milhas")

    # Critério Premium: preço > 50000 OU (preço > 40000 E ano >= 2021)
    df['premium'] = ((df['price'] > 50000) | ((df['price'] > 40000) & (df['year'] >= 2021))).astype(int)
    df['economico'] = 1 - df['premium']

    premium_count = df['premium'].sum()
    economico_count = len(df) - premium_count
    print(f"Carros Premium: {premium_count} ({premium_count / len(df) * 100:.1f}%)")
    print(f"Carros Econômicos: {economico_count} ({economico_count / len(df) * 100:.1f}%)")

    # Features: year, price, mileage
    X = df[['year', 'price', 'mileage']].values.astype(float)

    # Normalização manual (igual ao código original)
    X_norm = X.copy()

    # Normalizar Ano: min-max baseado nos dados
    year_min, year_max = X[:, 0].min(), X[:, 0].max()
    X_norm[:, 0] = (X[:, 0] - year_min) / (year_max - year_min)

    # Normalizar Preço: min-max baseado nos dados
    price_min, price_max = X[:, 1].min(), X[:, 1].max()
    X_norm[:, 1] = (X[:, 1] - price_min) / (price_max - price_min)

    # Normalizar Quilometragem: min-max baseado nos dados
    mileage_min, mileage_max = X[:, 2].min(), X[:, 2].max()
    X_norm[:, 2] = (X[:, 2] - mileage_min) / (mileage_max - mileage_min)

    # Adicionar bias
    bias = np.ones((X_norm.shape[0], 1))
    X_final = np.hstack([bias, X_norm])

    # Saídas: [premium, economico]
    y = df[['premium', 'economico']].values.astype(float)

    print(f"Dados preprocessados: {X_final.shape[0]} amostras, {X_final.shape[1]} features (com bias)")
    print(
        f"Ranges normalizados: Ano [{year_min}-{year_max}], Preço [${price_min:,}-${price_max:,}], KM [{mileage_min:,}-{mileage_max:,}]")

    return X_final, y, df


def sigmoid(x: float) -> float:
    """
    Função sigmoid para ativação
    """
    # Limitar x para evitar overflow
    x = np.clip(x, -500, 500)
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
    epocas = 1500
    taxa_aprendizado = 0.05

    print("=== PROBLEMA DE CLASSIFICAÇÃO: CARROS PREMIUM vs ECONÔMICO ===")
    print("\nCritério de classificação:")
    print("• Premium: Preço > $50,000 OU (Preço > $40,000 E Ano >= 2021)")
    print("• Econômico: Demais casos")
    print("\nCaracterísticas utilizadas: Ano, Preço, Quilometragem")
    print("\nArquitetura da rede MLP:")
    print("• Camada de entrada: 4 neurônios (bias + 3 características)")
    print("• Camada oculta: 5 neurônios com função sigmoid")
    print("• Camada de saída: 2 neurônios (Premium, Econômico)")
    print(f"• Taxa de aprendizado: {taxa_aprendizado}")
    print(f"• Épocas: {epocas}")

    # Carregar dados do CSV
    try:
        X, y, df_original = carregar_e_preprocessar_dados('cars.csv')
    except FileNotFoundError:
        print("\nErro: Arquivo 'cars.csv' não encontrado!")
        print("Certifique-se de que o arquivo está no mesmo diretório do script.")
        return
    except Exception as e:
        print(f"\nErro ao carregar dados: {e}")
        return

    # Dividir dados em treino e teste (80% treino, 20% teste)
    indices = np.arange(len(X))
    indices_treino, indices_teste = train_test_split(
        indices, test_size=0.2, random_state=42,
        stratify=y[:, 0]  # Estratificar baseado na coluna premium
    )

    print(f"\nDivisão dos dados:")
    print(f"Treino: {len(indices_treino)} amostras")
    print(f"Teste: {len(indices_teste)} amostras")

    # Inicialização dos pesos da rede
    # Camada Oculta: 5 neurônios com 4 entradas cada (bias + 3 características)
    pesos_camada_oculta = np.random.uniform(-0.5, 0.5, (5, 4))
    # Camada de Saída: 2 neurônios com 6 entradas cada (bias + 5 neurônios ocultos)
    pesos_camada_saida = np.random.uniform(-0.5, 0.5, (2, 6))

    print("\nIniciando treinamento da rede neural...")

    # Exemplo de cálculo manual para um padrão
    print("\n=== EXEMPLO DE PROPAGAÇÃO E RETROPROPAGAÇÃO ===")
    exemplo_idx = random.choice(indices_treino)  # Escolher um exemplo aleatório do treino
    linha_exemplo = X[exemplo_idx]
    carro_exemplo = df_original.iloc[exemplo_idx]
    print(f"Padrão escolhido: {carro_exemplo['make']} {carro_exemplo['model']} {carro_exemplo['year']}")
    print(f"Preço: ${carro_exemplo['price']:,}, Quilometragem: {carro_exemplo['mileage']:,}")
    print(f"Entrada normalizada: {linha_exemplo}")
    print(f"Classe real: {'Premium' if y[exemplo_idx][0] == 1 else 'Econômico'}")

    # Treinamento
    erro_por_epoca = []
    for epoca in range(epocas):
        random.shuffle(indices_treino)

        erro_total = 0  # Inicializa erro total da época

        for idx in indices_treino:
            linha = X[idx]
            saida_esperada = y[idx]

            # ===== PROPAGAÇÃO PARA FRENTE =====
            camada_oculta = np.zeros(5)
            for i in range(5):
                soma = 0
                for j in range(4):  # bias + 3 características
                    soma += linha[j] * pesos_camada_oculta[i][j]
                camada_oculta[i] = sigmoid(soma)

            camada_saida = np.zeros(2)
            for i in range(2):
                camada_saida[i] = pesos_camada_saida[i][0]  # Bias
                for j in range(1, 6):
                    camada_saida[i] += camada_oculta[j - 1] * pesos_camada_saida[i][j]

            # ===== ERRO DA AMOSTRA =====
            erro_amostra = np.sum((saida_esperada - camada_saida) ** 2)
            erro_total += erro_amostra

            # ===== RETROPROPAGAÇÃO =====
            delta_saida = np.zeros(2)
            for i in range(2):
                erro = saida_esperada[i] - camada_saida[i]
                delta_saida[i] = erro * 1  # Derivada linear

            delta_oculta = np.zeros(5)
            for i in range(5):
                soma = 0
                for j in range(2):
                    soma += delta_saida[j] * pesos_camada_saida[j][i + 1]
                delta_oculta[i] = camada_oculta[i] * (1 - camada_oculta[i]) * soma

            # Atualizar pesos - saída
            for i in range(2):
                pesos_camada_saida[i][0] += taxa_aprendizado * delta_saida[i]
                for j in range(1, 6):
                    pesos_camada_saida[i][j] += taxa_aprendizado * delta_saida[i] * camada_oculta[j - 1]

            # Atualizar pesos - oculta
            for i in range(5):
                pesos_camada_oculta[i][0] += taxa_aprendizado * delta_oculta[i]
                for j in range(1, 4):
                    pesos_camada_oculta[i][j] += taxa_aprendizado * delta_oculta[i] * linha[j]

        # ===== ERRO MÉDIO POR ÉPOCA =====
        erro_medio = erro_total / len(indices_treino)
        erro_por_epoca.append(erro_medio)

        if epoca % 200 == 0:
            print(f"Época: {epoca} concluída")

    print("Treinamento concluído!\n")

    #===== PLOTAGEM DO GRÁFICO =====
    plt.figure(figsize=(10, 4))
    plt.plot(erro_por_epoca, color='blue')
    plt.xlabel('Épocas')
    plt.ylabel('Erro Quadrático Médio')
    plt.title('Convergência do Erro durante o Treinamento')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('images/convergencia_erro.png')
    plt.show()

    # ===== TESTE DA REDE =====

    def testar_rede_com_metricas(indices: List[int], nome_conjunto: str) -> Dict:
        """Testa a rede com um conjunto de dados e calcula métricas"""
        print(f"=== RESULTADOS COM DADOS DE {nome_conjunto.upper()} ===")

        # Inicializar contadores da matriz de confusão
        tp = tn = fp = fn = 0

        for idx in indices:
            linha = X[idx]
            carro = df_original.iloc[idx]

            # Propagação para teste
            camada_oculta = np.zeros(5)
            for i in range(5):
                soma = 0
                for j in range(4):
                    soma += linha[j] * pesos_camada_oculta[i][j]
                camada_oculta[i] = sigmoid(soma)

            camada_saida = np.zeros(2)
            for i in range(2):
                camada_saida[i] = pesos_camada_saida[i][0]
                for j in range(1, 6):
                    camada_saida[i] += camada_oculta[j - 1] * pesos_camada_saida[i][j]

            # Normalizar saída para 0 ou 1
            if camada_saida[1] >= 0.5:
                camada_saida[1] = 1
            else:
                camada_saida[1] = 0
            if camada_saida[0] >= 0.5:
                camada_saida[0] = 1
            else:
                camada_saida[0] = 0

            # Determinar classes
            previsto_premium = camada_saida[0] > camada_saida[1]
            real_premium = y[idx][0] == 1

            classe_prevista = "PREMIUM" if previsto_premium else "ECONÔMICO"
            classe_real = "PREMIUM" if real_premium else "ECONÔMICO"

            # Calcular matriz de confusão
            if previsto_premium and real_premium:
                tp += 1
                resultado = "(TP)"
            elif not previsto_premium and not real_premium:
                tn += 1
                resultado = "(TN)"
            elif previsto_premium and not real_premium:
                fp += 1
                resultado = "(FP)"
            else:
                fn += 1
                resultado = "(FN)"

            # Mostrar apenas primeiros 10 para não poluir a saída
            if len([i for i in indices if i <= idx]) <= 10:
                # Garante que as saídas sejam 0 ou 1 (não negativas)
                saida_binaria = [1 if v > 0 else 0 for v in camada_saida]
                print(f"{carro['make']:>14} {carro['model']:<16} {carro['year']:<4.0f}: "
                      f"${carro['price']:<7,.0f}, {carro['mileage']:<6,.0f}mi "
                      f"-> {classe_prevista:<9s} [{saida_binaria[0]}, {saida_binaria[1]}] {resultado}")

        total = len(indices)
        acertos = tp + tn
        print(f"... (mostrando apenas primeiros 10 resultados)")
        print(f"Acertos no {nome_conjunto}: {acertos}/{total} ({acertos / total * 100:.1f}%)")

        # Calcular métricas
        metricas = calcular_metricas(tp, tn, fp, fn)

        return {
            'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
            'acertos': acertos, 'total': total,
            'metricas': metricas
        }

    # Teste com dados de treino
    resultado_treino = testar_rede_com_metricas(indices_treino[:50], "treino")  # Primeiros 50 para não poluir
    print()

    # Teste com dados de teste
    resultado_teste = testar_rede_com_metricas(indices_teste, "teste")
    print()

    # ===== MATRIZ DE CONFUSÃO E MÉTRICAS =====
    print("=== MATRIZ DE CONFUSÃO - DADOS DE TESTE ===")
    tp, tn, fp, fn = resultado_teste['tp'], resultado_teste['tn'], resultado_teste['fp'], resultado_teste['fn']
    print("                        PREVISTO")
    print("               ┌─────────────┬─────────────┐")
    print("               │   Premium   │  Econômico  │")
    print("R   ┌──────────┼─────────────┼─────────────┤")
    print(f"E   │ Premium  │   {tp:^9} │   {fn:^9} │")
    print("E   ├──────────┼─────────────┼─────────────┤")
    print(f"A   │Econômico │   {fp:^9} │   {tn:^9} │")
    print("L   └──────────┴─────────────┴─────────────┘\n")

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
    print(f"\n🎯 AVALIAÇÃO GERAL DA REDE:")
    if accuracy_teste > 85:
        print("   🎉 Rede generalizando MUITO BEM para classificação de carros!")
        print("   📈 Excelente capacidade de predição em dados não vistos.")
    elif accuracy_teste > 75:
        print("   ✅ Rede com BOM DESEMPENHO na classificação.")
        print("   📊 Boa capacidade de generalização.")
    elif accuracy_teste > 60:
        print("   ⚠️  Rede com desempenho MODERADO.")
        print("   🔄 Pode necessitar ajustes nos hiperparâmetros.")
    else:
        print("   ❌ Rede com possível OVERFITTING.")
        print("   🔧 Recomenda-se revisar arquitetura e dados de treino.")

    # ===== ANÁLISE DOS CRITÉRIOS E DADOS =====
    print(f"\n╔═══════════════════════════════════════════════════════════╗")
    print(f"║              CRITÉRIOS DE MERCADO UTILIZADOS              ║")
    print(f"╚═══════════════════════════════════════════════════════════╝")

    print(f"\n🏷️  CLASSIFICAÇÃO PREMIUM vs ECONÔMICO:")
    print(f"   \n   🏆 PREMIUM:")
    print(f"      • Veículos de luxo, alta tecnologia, ou alto valor")
    print(f"      • Critério: Preço > $50,000")
    print(f"      • OU (Preço > $40,000 E Ano >= 2021)")
    print(f"      • Marcas típicas: BMW, Mercedes, Audi, Porsche, Tesla")
    print(f"   \n   🚗 ECONÔMICO:")
    print(f"      • Veículos de entrada, foco em custo-benefício")
    print(f"      • Preço ≤ $40,000 ou modelos mais antigos")
    print(f"      • Ênfase em economia e praticidade")

    print(f"\n🔍 FATORES CONSIDERADOS NA CLASSIFICAÇÃO:")
    print(f"   • 📅 Ano:           Carros mais novos tendem a ser mais valorizados")
    print(f"   • 💰 Preço:         Principal indicador de categoria de mercado")
    print(f"   • 🛣️  Quilometragem: Afeta valor de revenda e percepção de qualidade")

    print(f"\n📋 EXEMPLOS DO DATASET:")

    # Mostrar alguns exemplos de cada classe
    premium_examples = df_original[df_original['premium'] == 1].head(3)
    economico_examples = df_original[df_original['premium'] == 0].head(3)

    print(f"\n   🏆 CARROS PREMIUM (Exemplos):")
    for _, car in premium_examples.iterrows():
        print(
            f"      • {car['make']:12} {car['model']:15} {car['year']} - ${car['price']:6,} ({car['mileage']:,} milhas)")

    print(f"\n   🚗 CARROS ECONÔMICOS (Exemplos):")
    for _, car in economico_examples.iterrows():
        print(
            f"      • {car['make']:12} {car['model']:15} {car['year']} - ${car['price']:6,} ({car['mileage']:,} milhas)")

    print(f"\n╔═══════════════════════════════════════════════════════════╗")
    print(f"║                      CONCLUSÃO                            ║")
    print(f"╚═══════════════════════════════════════════════════════════╝")
    print(f"\n✨ A rede neural MLP foi treinada com sucesso para classificar")
    print(f"   carros em categorias Premium e Econômico, utilizando apenas")
    print(f"   três características: ano, preço e quilometragem.")
    print(f"\n🎓 Este modelo pode ser útil para:")
    print(f"   • Concessionárias organizarem inventário")
    print(f"   • Sistemas de recomendação de veículos")
    print(f"   • Análise automatizada de mercado automotivo")
    print(f"   • Precificação inteligente de veículos")


if __name__ == "__main__":
    # Seed para reprodutibilidade
    np.random.seed(42)
    random.seed(42)

    main()
