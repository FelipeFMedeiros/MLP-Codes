# 🧠 Classificação de Carros com MLP (Perceptron Multi-Camadas)

Este projeto implementa uma Rede Neural Artificial do tipo Perceptron Multi-Camadas (MLP) para resolver um problema de classificação de padrões: **classificar carros em categorias Premium e Econômico** com base em características como ano, preço e quilometragem.

## 📊 Problema de Classificação

**Objetivo:** Desenvolver um sistema de classificação automática que ajude concessionárias a categorizar veículos de forma inteligente, facilitando organização de inventário, precificação e recomendações aos clientes.

**Critérios de Classificação:**
- **Premium:** Veículos com preço > $50.000 OU (preço > $40.000 E ano >= 2021)
- **Econômico:** Demais veículos

**Características Utilizadas:**
- Ano do veículo
- Preço de venda 
- Quilometragem 

⚠️ Os dados são normalizados para melhorar a eficiência e estabilidade da rede neural.

## 📌 Justificativa e Relevância

O setor automotivo frequentemente lida com grandes volumes de dados sobre veículos. Classificações manuais são subjetivas, demoradas e propensas a erro humano. Usar aprendizado de máquina para este fim:

* Reduz custos operacionais

* Aumenta a eficiência de vendas

* Oferece suporte à precificação e segmentação de clientes

## ⚙️ Arquitetura da Rede MLP

A rede neural implementada possui a seguinte arquitetura:

- **Camada de Entrada:** 4 neurônios (1 bias + 3 características normalizadas)
- **Camada Oculta:** 5 neurônios com função de ativação sigmoid
- **Camada de Saída:** 2 neurônios (saída linear para classes Premium/Econômico)
- **Algoritmo de Treinamento:** Backpropagation com taxa de aprendizado de 0.05
- **Épocas de Treinamento:** 1.500

## 📊 Dados Utilizados

* Fonte: Arquivo `cars.csv`

* Colunas usadas:

   * `year`, `price`, `mileage`

* Target: `[premium, economico]` (valores binários)

Normalização

Todos os atributos são normalizados via min-max scaling:

```Python
X_norm = (X - X.min()) / (X.max() - X.min())
```

## 🧪 Saída da Rede Neural: Como Interpretar?

A camada de saída possui 2 neurônios, correspondendo às classes:

* Índice 0 → Premium

* Índice 1 → Econômico

A saída pode gerar valores contínuos como [0.8, 0.2]. Para fins de avaliação, esses valores são normalizados para 0 ou 1 da seguinte forma:

```python
if output[i] >= 0.5:
    output[i] = 1
else:
    output[i] = 0
```

## 🚦 Interpretação das Saídas
| 🔢 **Saída da Rede** | 🏷️ **Classe Interpretada** | 💬 **Observações**                                       |
| -------------------- | --------------------------- | -------------------------------------------------------- |
| `[1, 0]`             | **Premium**                 | Correta — classe Premium foi corretamente identificada   |
| `[0, 1]`             | **Econômico**               | Correta — classe Econômico foi corretamente identificada |
| `[1, 1]`             | **Premium**                 | Ambígua — valores iguais, favorece Premium (índice 0)    |
| `[0, 0]`             | **Econômico**               | Ambígua — valores iguais, favorece Econômico (índice 1)  |

### 🔍 Nota: Nos casos ambíguos ([1,1] ou [0,0]), a decisão é tomada comparando os dois neurônios de saída:

* Se output[0] > output[1] → classificado como Premium

* Se output[1] > output[0] → classificado como Econômico

## 📊 Métricas de Desempenho

Após o treinamento e validação do modelo MLP, as seguintes métricas foram calculadas com base nos dados de teste:

### 🧮 Matriz de Confusão
```
                     PREVISTO
               ┌─────────────┬─────────────┐
               │  Premium    │  Econômico  │
───────────────┼─────────────┼─────────────┤
REAL  Premium  │     TP      │     FN      │
      Econômico│     FP      │     TN      │
               └─────────────┴─────────────┘
```

| Símbolo | Significado                                          |
| ------- | ---------------------------------------------------- |
| TP      | True Positive – Premium corretamente classificado    |
| TN      | True Negative – Econômico corretamente classificado  |
| FP      | False Positive – Econômico classificado como Premium |
| FN      | False Negative – Premium classificado como Econômico |


### ✅ Métricas Calculadas

As fórmulas utilizadas e seus significados são listados abaixo, seguidas dos valores calculados:

| 📈 **Métrica** | 🧮 **Fórmula**                                  | 📖 **Descrição**                                                             |
| -------------- | ----------------------------------------------- | ---------------------------------------------------------------------------- |
| **Acurácia**   | `(TP + TN) / (TP + TN + FP + FN)`               | Percentual total de acertos                                                  |
| **Precisão**   | `TP / (TP + FP)`                                | Entre os classificados como Premium, quantos realmente são Premium           |
| **Recall**     | `TP / (TP + FN)`                                | Entre os que são realmente Premium, quantos foram corretamente classificados |
| **F1-Score**   | `2 * (Precisão * Recall) / (Precisão + Recall)` | Média harmônica entre Precisão e Recall                                      |

### 📋 Exemplo de Resultados (Dados de Teste)
| 📌 **Métrica** | 📊 **Valor** |
| -------------- | ------------ |
| **Acurácia**   | `92.5%`      |
| **Precisão**   | `93.8%`      |
| **Recall**     | `88.2%`      |
| **F1-Score**   | `90.9%`      |


#### 📌 Interpretação:

* O modelo acerta mais de 9 em cada 10 classificações.
* A alta precisão indica que quase todos os carros classificados como Premium realmente pertencem a essa classe.
* O bom recall mostra que a maior parte dos carros Premium foram corretamente identificados.
* O F1-score elevado confirma equilíbrio entre precisão e recall — ideal para problemas com impacto prático nas decisões (ex: precificação, recomendação de veículos).

## ✅ Funcionalidades Implementadas

✅ **Definição e descrição do problema de classificação**  
✅ **Arquitetura da rede MLP detalhada**  
✅ **Cálculo de propagação e retropropagação** (demonstrado com exemplo aleatório)  
✅ **Treinamento completo com backpropagation**  
✅ **Matriz de confusão e métricas de desempenho** (acurácia, precisão, recall, F1-score)

## Estrutura do Projeto

```
mlp_car_classification/
├── mlp_car_classifier.py    # Script principal com implementação completa da MLP
├── cars.csv                 # Conjunto de dados de carros
├── requirements.txt         # Dependências Python
└── README.md               # Esta documentação
```

## Dependências

As seguintes bibliotecas são necessárias (instale com `pip install -r requirements.txt`):

- numpy
- pandas
- scikit-learn

## Como Executar

1. **Instalar dependências:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Executar o classificador:**
   ```bash
   python mlp_car_classifier.py
   ```

O script irá:
- Carregar e pré-processar os dados do `cars.csv`
- Treinar a rede neural por 1.500 épocas
- Mostrar exemplo detalhado de propagação e retropropagação
- Avaliar o desempenho com matriz de confusão e métricas
- Apresentar análise completa dos resultados

## Exemplo de Saída

```
=== PROBLEMA DE CLASSIFICAÇÃO: CARROS PREMIUM vs ECONÔMICO ===

Critério de classificação:
• Premium: Preço > $50,000 OU (Preço > $40,000 E Ano >= 2021)
• Econômico: Demais casos

Arquitetura da rede MLP:
• Camada de entrada: 4 neurônios (bias + 3 características)
• Camada oculta: 5 neurônios com função sigmoid
• Camada de saída: 2 neurônios (Premium, Econômico)

=== EXEMPLO DE PROPAGAÇÃO E RETROPROPAGAÇÃO ===
Padrão escolhido: BMW i4 2023
Preço: $56,000, Quilometragem: 3,000
Classe real: Premium

=== RESULTADOS COM DADOS DE TESTE ===
BMW           i4               2023: $56,000, 3,000mi -> PREMIUM    [1, 0] (TP)
Mercedes      C-Class          2022: $45,000, 15,000mi -> PREMIUM    [1, 0] (TP)
...

=== MATRIZ DE CONFUSÃO - DADOS DE TESTE ===
                        PREVISTO
               ┌─────────────┬─────────────┐
               │   Premium   │  Econômico  │
R   ┌──────────┼─────────────┼─────────────┤
E   │ Premium  │     15      │      2      │
E   ├──────────┼─────────────┼─────────────┤
A   │Econômico │      1      │     22      │
L   └──────────┴─────────────┴─────────────┘

=== MÉTRICAS DETALHADAS - DADOS DE TESTE ===
Acurácia = 92.5%
Precisão = 93.8%
Recall = 88.2%
F1-score = 90.9%
```

## Aplicações Práticas

Este modelo pode ser utilizado para:
- **Concessionárias:** Organização automática de inventário por categoria
- **Sistemas de recomendação:** Sugestão de veículos baseada no perfil do cliente
- **Precificação:** Ajuste automático de preços baseado na categoria
- **Análise de mercado:** Identificação de tendências em vendas de veículos

## Conclusão

A implementação demonstra com sucesso o uso de Redes Neurais Artificiais para classificação de padrões reais, especificamente no contexto automotivo. O modelo alcança bom desempenho na distinção entre carros Premium e Econômicos, validando a eficácia da arquitetura MLP proposta para este tipo de problema de classificação binária.
