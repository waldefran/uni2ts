Apresento o **Plano de Projeto 2.0**, em ordem cronológica, da criação do cérebro à sua operação em tempo real.

---

### **FASE 1: FINE-TUNING (A Criação do Cérebro Especialista)**

**1.1 - Preparação do Super-Dataset de Treinamento (Upgrade de Filosofia)**
* **Ação:** Esqueça a ideia de tratar os ativos como canais separados. Conforme nossa discussão, vamos adotar a abordagem **univariada, unificada e anônima.**
    * Coletaremos os últimos 3-4 anos de dados de M1 para os 2 principais ativos de Cripto.
    * **Criação das Amostras:** Para cada ativo, geraremos milhares de amostras. Cada amostra é uma sequência contínua de 2048 candles de M1. 
    * **Unificação e Embaralhamento:** Todas as amostras de todos os ativos serão unificadas em um único "Super-Dataset" e rigorosamente embaralhadas. O modelo não saberá a qual ativo cada amostra pertence.

*Dados*

| Posição no Array | Campo (em inglês) | Descrição | Exemplo |
| --- | --- | --- | --- |
| Open Time | Timestamp de abertura do candle (milissegundos) | 1678886400000 |
| Open | Preço de abertura | 24000.00 |
| High | Preço máximo | 24100.00 |
| Low | Preço mínimo | 23900.00 |
| Close | Preço de fechamento | 24050.00 |
| Volume | Volume do ativo base | 100.50 |
| Close Time | Timestamp de fechamento do candle (milissegundos) | 1678886459999 |
| Quote Asset Volume | Volume do ativo de cotação | 2417025.00 |
| **Number of Trades** | **Quantidade de negócios realizados** | **500** |
| **Taker Buy Base Asset Volume** | **Volume do ativo base comprado por "takers"** | **50.25** |
| **Taker Buy Quote Asset Volume** | **Volume do ativo de cotação comprado por "takers"**| **1208512.50**|


**1.2 - Pré-processamento e Normalização (Upgrade de Técnica)**
* **Ação:** Para cada amostra de 2048 candles no dataset, aplicaremos:
    1.  **Normalização pela Janela:** Todos os valores serão normalizados em relação ao primeiro passo de tempo da janela (ex: `valor_norm = (valor[t] / valor[t=0]) - 1`). Isso remove a escala absoluta e foca o modelo na forma do padrão.
    2.  **Inclusão de Features Temporais Cíclicas (Upgrade da Máscara de Mercado):** Em vez de uma máscara booleana, vamos adicionar features que permitem ao modelo *aprender* os ciclos de mercado. Para cada um dos 2048 passos, adicionaremos:
        * `minuto_da_hora (sin/cos)`
        * `hora_do_dia (sin/cos)`
        * `dia_da_semana (sin/cos)`
        * A transformação seno/cosseno ajuda o modelo a entender a natureza cíclica do tempo (ex: a hora 23 é próxima da hora 0). O modelo aprenderá que os padrões nos `dias_da_semana` 5 e 6 (Sáb/Dom) são diferentes dos outros.
* **Resultado:** Cada amostra de treino terá a dimensão `[2048, x]` 

**1.3 - Treinamento e Função de Perda (Upgrade de Otimização)**
* **Ação:** Durante o fine-tuning, o modelo aprenderá a prever uma **sequência futura** de passos de tempo, não apenas 4 pontos fixos.
* **Rótulo (Label):** Para cada amostra de entrada `[t-2048, ..., t-1]`, o rótulo será a sequência real dos próximos 60 candles de M1: `[t, t+1, ..., t+59]`.
* **Função de Perda (Loss Function - Upgrade para Probabilística):** Em vez de uma perda ponderada com MSE/MAE, usaremos a **Negative Log-Likelihood (NLL)**. Como nossa cabeça Bayesiana irá prever uma distribuição (média e variância) para cada futuro passo de tempo, a NLL é a função de perda matematicamente correta. Ela otimiza o modelo para produzir distribuições de probabilidade precisas, penalizando-o severamente não apenas por errar a média, mas por estar muito confiante (baixa variância) quando erra.
* **Resultado do Fine-Tuning:** Um único arquivo de modelo, `moirai_v2_finetuned.pth`, contendo um cérebro especialista em prever a próxima hora de dados de M1 para qualquer um dos ativos.

---

### **FASE 2: A PIPELINE DE INFERÊNCIA EM TEMPO REAL (O Sistema em Operação 24/7)**

Este é o fluxo executado a cada minuto na instância com GPU T4.

**Módulo 1: Pré-processamento 2.0 (Da Realidade ao Tensor)**
* **Gatilho:** Executado uma vez por minuto, no fechamento do candle de M1.
* **Processo:**
    1.  **Aquisição:** Busca o último candle de M1 para os ativos.
    2.  **Construção do Batch:** Para cada ativo, o sistema monta sua janela de contexto de 2048 candles de M1.
    3.  **Normalização e Feature Engineering:** O mesmo processo da Fase 1.2 (Normalização pela Janela + Features Temporais) é aplicado a cada uma das janelas de contexto.
    4.  **Tensorização Final (Upgrade de Arquitetura):** Os tensores de amostra são empilhados para formar um único batch.
        * **Dimensões:** `[batch_size, sequence_length, num_features]`
        * **Valores Concretos:** `[2, 2048, x]`
        * `batch_size = 2`: As séries de ativos são processadas em paralelo pela GPU, otimizando o uso do hardware. Cada uma é tratada como um exemplo independente no batch.

**Módulo 2: Motor de Inferência 2.0 (O Coração Otimizado)**
* **Carregamento:** O modelo `moirai_v2_finetuned.pth` é carregado na GPU e compilado com **NVIDIA TensorRT** para máxima performance.
* **Processo de Previsão (Upgrade para Sequência Única):**
    1.  O batch `[2, 2048, x]` é alimentado no modelo.
    2.  Ocorre um único *forward pass* pelo corpo do Transformer.
    3.  Uma **única cabeça de previsão Bayesiana** gera a saída.
    4.  **Resultado Bruto:** A saída é uma sequência de distribuições de probabilidade para os próximos 60 minutos. A dimensão da saída será `[2, 60, x]` (ativos, 60 passos de tempo futuros, valores para média, variância da distribuição, etc...). A técnica de MC Dropout é aplicada apenas nesta cabeça para gerar as amostras de incerteza.

**Módulo 3: Pós-processamento 2.0 (Do Tensor à Inteligência Acionável)**
* **Ação:** Este módulo recebe a previsão de sequência e a "descompacta" em insights para cada um dos ativos.
* **Derivação dos Timeframes:** A partir da previsão para os próximos 60 minutos, ele deriva as previsões para os horizontes desejados:
    * **M1:** Usa a previsão para `t+1`.
    * **M5:** Usa a previsão para `t+5`.
    * **M15:** Usa a previsão para `t+15`.
    * **H1:** Usa a previsão para `t+60`.
* **Cálculo de Estatísticas Avançadas (Upgrade de Métricas):** Para cada um desses pontos (M1, M5, M15, H1), o sistema calcula:
    * `mean_prediction`: A previsão de ponto (média da distribuição).
    * `uncertainty_std`: O desvio padrão (a incerteza).
    * `prediction_quantiles`: Os valores nos percentis 10%, 50% (mediana) e 90%, para entender a assimetria do risco.
    * `breakout_probability`: A probabilidade calculada da distribuição de o preço exceder um limiar importante (ex: `preço_atual + X%`).
* **Saída Final:** Um objeto JSON claro, porém agora enriquecido com as métricas avançadas, pronto para ser consumido pelo módulo de lógica de trading.
**ESSE outro MÓDULO É UM OUTRO MODELO. Ele será treinado com algoritimos de Reinforcement Learn.**
Aqui apenas fazemos a previsão.


Este Plano 2.0 é robusto, alinhado com a filosofia de modelos de fundação e otimizado para extrair o máximo de informação e performance, representando um verdadeiro roteiro para um sistema estado da arte.


Quando em produção rodara em AWS com GPU A100/H100.
Durante DEV e Fine Tuning será no Kaggle(com GPU) para POC, onde transformamos o Moirai-MoE genérico em um especialista no nosso portfólio.

Obs: agora estamos em um notebook pessoal, Dell Core i5 com 8Gb de RAM e sem placa de vídeo.
Todo o código que fizermos aqui, será executado preferencialmente em um notebook no kaggle ou colab para performace em GPU visando o fine tuning.