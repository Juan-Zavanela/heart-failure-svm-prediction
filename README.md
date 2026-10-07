# Previsão de Doença Cardíaca com SVM

Projeto acadêmico de treinamento de um modelo de Support Vector Machine (SVM) para prever a presença de doença cardíaca em pacientes, usando o [Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction) (Kaggle).

## Objetivo

Classificar se um paciente possui ou não doença cardíaca (`HeartDisease`: 0 = sem doença, 1 = com doença) a partir de variáveis clínicas como idade, sexo, tipo de dor no peito, pressão arterial em repouso, colesterol, entre outras.

## Pipeline

1. **Análise exploratória**: distribuição da variável alvo, distribuição das variáveis numéricas, matriz de correlação e identificação de outliers/valores suspeitos.
2. **Tratamento de dados ausentes**: `RestingBP` e `Cholesterol` continham zeros clinicamente impossíveis, tratados como dados ausentes e substituídos pela mediana da coluna.
3. **Encoding**:
   - Variáveis binárias (`Sex`, `ExerciseAngina`) → mapeadas para 0/1.
   - Variável ordinal (`ST_Slope`) → mapeada respeitando a ordem clínica (0/1/2).
   - Variáveis categóricas sem ordem (`ChestPainType`, `RestingECG`) → One-Hot Encoding.
4. **Preparação para o treino**: padronização das variáveis com `StandardScaler` (essencial para SVM, que trabalha com distância entre pontos) e separação treino/teste (80/20) com `stratify=y`.
5. **Treinamento**: busca dos melhores hiperparâmetros (`kernel`, `C`, `gamma`) via `GridSearchCV` com validação cruzada (cv=5).
6. **Avaliação**: acurácia, relatório de classificação (precision/recall/f1) e matriz de confusão no conjunto de teste.

## Resultados

- **Melhores hiperparâmetros**: `kernel=linear`, `C=1`, `gamma=scale`
- **Acurácia média na validação cruzada**: 85,4%
- **Acurácia no conjunto de teste**: 86,96%

| Classe | Precision | Recall | F1-score |
|---|---|---|---|
| Sem doença (0) | 0.88 | 0.82 | 0.85 |
| Com doença (1) | 0.86 | 0.91 | 0.89 |

## Tecnologias

- Python
- pandas, numpy
- scikit-learn (SVM, GridSearchCV, métricas)
- matplotlib, seaborn (visualização)

## Como executar

1. Clone o repositório e instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```
2. O dataset já está incluído em `data/heart.csv` (fonte: [Heart Failure Prediction Dataset, Kaggle](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)).
3. Abra [`notebook/heart-failure-svm.ipynb`](notebook/heart-failure-svm.ipynb) em um ambiente Jupyter e execute as células na ordem.

## Autores

## Autores

Projeto acadêmico desenvolvido em dupla na UMC.

## Autores

Projeto acadêmico desenvolvido em dupla na UMC.

- **Juan Zavanela**: [LinkedIn](https://www.linkedin.com/in/juan-zavanela-32a90134b) · [GitHub](https://github.com/Juan-Zavanela)
- **Jonathan Simões**: [LinkedIn](https://www.linkedin.com/in/jonathansim%C3%B5esjs/)
