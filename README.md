# 🏥 MLP_ClassificarDiabete

Este projeto implementa uma **Rede Neural Multicamada (MLP - Multilayer Perceptron)** para classificação de diabetes baseada em dados clínicos. O sistema utiliza duas características principais: **IMC (Índice de Massa Corporal)** e **Glicemia** para prever se um paciente possui diabetes ou não.

## 🎯 Objetivo

Desenvolver um sistema de classificação binária que possa auxiliar na detecção precoce de diabetes, utilizando uma abordagem de aprendizado supervisionado com redes neurais artificiais.

## 🏗️ Arquitetura da Rede

-   **Camada de Entrada**: 3 neurônios (bias, IMC, glicemia)
-   **Camada Oculta**: 3 neurônios com função de ativação sigmoid
-   **Camada de Saída**: 2 neurônios (com diabetes, sem diabetes)
-   **Algoritmo de Treinamento**: Retropropagação (Backpropagation)

## 📊 Dataset

O projeto utiliza um conjunto de 16 amostras sintéticas com as seguintes características:

| Característica | Faixa de Valores | Descrição                |
| -------------- | ---------------- | ------------------------ |
| **IMC**        | 16 - 40 kg/m²    | Índice de Massa Corporal |
| **Glicemia**   | 70 - 126 mg/dL   | Glicemia em jejum        |

### Divisão dos Dados

-   **Treinamento**: 10 amostras (62.5%)
-   **Teste**: 6 amostras (37.5%)

## ⚙️ Parâmetros da Rede

-   **Épocas**: 1.000
-   **Taxa de Aprendizado**: 0.1
-   **Função de Ativação (Oculta)**: Sigmoid
-   **Função de Ativação (Saída)**: Linear
-   **Normalização**: Min-Max Scaling (0-1)

## 🔍 Critérios Clínicos

### IMC (Índice de Massa Corporal)

-   **< 18,5**: Baixo peso
-   **18,5 - 24,9**: Peso normal
-   **25,0 - 29,9**: Sobrepeso
-   **≥ 30,0**: Obesidade

### Glicemia em Jejum

-   **≤ 100 mg/dL**: Normal
-   **100 - 125 mg/dL**: Pré-diabetes
-   **≥ 126 mg/dL**: Diabetes

## 📈 Métricas de Avaliação

O sistema calcula as seguintes métricas de performance:

-   **Acurácia**: Proporção de predições corretas
-   **Precisão**: Proporção de casos positivos corretamente identificados
-   **Recall (Sensibilidade)**: Proporção de casos positivos detectados
-   **F1-Score**: Média harmônica entre precisão e recall

## 🚀 Como Executar

### Pré-requisitos

```bash
pip install numpy
```

### Execução

```bash
python mlp_diabetes.py
```

## 🧠 Funcionalidades

### Principais Funções

-   **`normalizar_dados()`**: Normaliza os dados de entrada entre 0 e 1
-   **`sigmoid()`**: Função de ativação sigmoid
-   **`calcular_metricas()`**: Calcula métricas de avaliação (acurácia, precisão, recall, F1-score)
-   **`testar_rede_com_metricas()`**: Testa a rede e gera relatório detalhado

### Processo de Treinamento

1. **Inicialização**: Pesos aleatórios entre -1 e 1
2. **Propagação Direta**: Cálculo das saídas das camadas
3. **Cálculo do Erro**: Diferença entre saída esperada e obtida
4. **Retropropagação**: Ajuste dos pesos baseado no erro
5. **Repetição**: Por 1.000 épocas

## 📊 Exemplo de Saída

```
=== RESULTADOS COM DADOS DE TESTE ===
Paciente  1:    IMC=25,     Glic=70     ->  SEM DIABETES    [0, 1]  (TN)
Paciente  2:    IMC=20,     Glic=72     ->  SEM DIABETES    [0, 1]  (TN)
...

=== MÉTRICAS DETALHADAS - DADOS DE TESTE ===
Acurácia = (TP + TN) / (TP + TN + FP + FN) = 83.33%
Precisão = TP / (TP + FP) = 100.00%
Recall = TP / (TP + FN) = 66.67%
F1-score = 2 * (Precisão * Recall) / (Precisão + Recall) = 80.00%
```

## 🎓 Contexto Acadêmico

Este projeto foi desenvolvido como parte dos estudos em **Redes Neurais Artificiais**.

---

\*Projeto desenvolvido para fins educacionais - Classificação de Diabetes com MLP*
