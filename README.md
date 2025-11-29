# KNN — Classificação de Sintomas (simples)

Projeto simples que treina e avalia um classificador K-Nearest Neighbors (KNN) para distinguir entre COVID-19, Gripe e Resfriado usando um dataset convertido para valores numéricos.

## O que este projeto faz

- Lê o arquivo `large_data_convertido.csv` (base numérica de sintomas).
- Treina um classificador KNN com K=3,5,7.
- Mostra as métricas (acurácia, precisão, recall, F1) no terminal.
- Salva as matrizes de confusão como imagens `matriz_confusao_k_3.png`, `matriz_confusao_k_5.png`, `matriz_confusao_k_7.png`.

## Requisitos

- Python 3.8+ instalado.
- Arquivos no mesmo diretório: `main.py`, `large_data_convertido.csv`, `requirements.txt` (opcional).

## Como rodar (Windows — PowerShell)

1. Abra o PowerShell na pasta do projeto (`c:\Users\Meu Computador\Desktop\knn_gripe`).

2. (Opcional, recomendado) Crie e ative um ambiente virtual, e instale dependências:

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

3. Execute o script:

```powershell
python main.py
```

4. O que você verá/gerará:
- Saída no terminal com os blocos "--- Resultados para k=3/5/7 ---" mostrando Acurácia, Precisão, Recall e F1-Score.
- Arquivos de imagem gerados no diretório: `matriz_confusao_k_3.png`, `matriz_confusao_k_5.png`, `matriz_confusao_k_7.png`.

## Autores

- Bruno Lopes
- Marcelo Augusto 
---