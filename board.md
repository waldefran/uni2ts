### **Roadmap de Implementação Incremental: Moirai Moe SOTA para Cripto**

#### **Fase 1: A Fundação dos Dados e o Cérebro Bayesiano (Semanas 1-3)**

**Objetivo:** Ter um dataset de altíssima qualidade pronto para uso e todas as classes Python customizadas que definem a "inteligência" do modelo. Ao final desta fase, temos os ingredientes, mas ainda não assamos o bolo.

*   **Semana 1: Aquisição e Exploração de Dados Brutos**
    *   **Atividade Principal:** Implementar o coletor de dados da Binance.
    *   **Arquivo(s) Chave:** `scripts/collect_data.py`, `src/crypto_data/binance_collector.py`.
    *   **Passos:**
        1.  Configurar a conexão com a API da Binance.
        2.  Implementar a lógica para baixar o histórico de candles de M1 para múltiplos ativos (BTC, ETH, etc.), salvando em um formato bruto (ex: Parquet).
        3.  Executar o script para obter os últimos 3-4 anos de dados.
        4.  Criar o notebook `notebooks/data_exploration.ipynb` para uma análise inicial dos dados brutos (verificar gaps, anomalias, etc.).
    *   **Entrega:** Uma pasta local `raw_data/` com todos os dados históricos necessários, validada.

*   **Semana 2: Construção do Super-Dataset de Treinamento (SOTA)**
    *   **Atividade Principal:** Implementar o `Data Builder` que transforma dados brutos em um dataset unificado e anônimo.
    *   **Arquivo(s) Chave:** `src/uni2ts/data/builder/crypto.py`, `scripts/prepare_dataset.py`.
    *   **Passos:**
        1.  Implementar a classe `CryptoDatasetBuilder`.
        2.  Dentro dela, implementar a lógica de **unificação** (juntar todos os ativos).
        3.  Implementar a **normalização por janela** `(valor[t] / valor[t=0]) - 1`.
        4.  Implementar a adição de **features cíclicas explícitas** (sin/cos).
        5.  Garantir a **anonimização** (remoção do `item_id`) para o conjunto de treino.
        6.  Executar `scripts/prepare_dataset.py` para gerar a versão final do dataset.
    *   **Entrega:** Uma pasta `crypto_dataset/` contendo o dataset processado no formato Hugging Face, pronto para o treinamento.

*   **Semana 3: Codificação do Núcleo Bayesiano**
    *   **Atividade Principal:** Criar as classes PyTorch customizadas que formam o cérebro Bayesiano do modelo.
    *   **Arquivo(s) Chave:** `src/uni2ts/model/crypto/bayesian_head.py`, `src/uni2ts/loss/bayesian_elbo.py`.
    *   **Passos:**
        1.  Implementar a classe `BayesianLinear` com seus parâmetros variacionais.
        2.  Implementar a classe `BayesianPredictionHead` completa, que utiliza a `BayesianLinear`, MC Dropout e a projeção para os parâmetros da distribuição Student-T.
        3.  Implementar a função de perda `BayesianELBOLoss`.
    *   **Entrega:** Módulos Python testáveis contendo as classes que são o coração da quantificação de incerteza do projeto.

---

#### **Fase 2: Treinamento e Geração do Modelo (Semanas 4-6)**

**Objetivo:** Utilizar os componentes da Fase 1 para treinar o Moirai-MoE e gerar o primeiro artefato de inteligência: o arquivo de modelo (`.ckpt`).

*   **Semana 4: Configuração e Pipeline de Treinamento**
    *   **Atividade Principal:** Conectar todas as peças: dados, modelo e configuração de treinamento.
    *   **Arquivo(s) Chave:** `configs/crypto_finetune.yaml`, `scripts/finetune_model.py`.
    *   **Passos:**
        1.  Criar o arquivo de configuração `crypto_finetune.yaml`, definindo o uso do Moirai-MoE, os caminhos para o dataset unificado, e todos os hiperparâmetros (batch size, learning rate, `kl_weight`, etc.).
        2.  Preparar o ambiente de treinamento em nuvem (Kaggle/Colab com GPU), fazendo o upload do dataset da Semana 2.
        3.  Implementar callbacks essenciais como `ModelCheckpoint` e `EarlyStopping` monitorando `val/ELBO`.
    *   **Entrega:** Um ambiente de treinamento completamente configurado, pronto para iniciar o fine-tuning com um único comando.

*   **Semanas 5-6: Execução do Fine-Tuning e Análise Inicial**
    *   **Atividade Principal:** Rodar o processo de fine-tuning e obter o modelo treinado.
    *   **Passos:**
        1.  Iniciar o treinamento na nuvem.
        2.  Monitorar ativamente o processo, observando a curva de perda `val/ELBO`.
        3.  Após a conclusão, baixar os checkpoints do modelo treinado (`.ckpt`) para o ambiente local.
        4.  Criar o notebook `notebooks/model_validation.ipynb` para carregar o modelo e fazer uma primeira previsão "bruta" em dados de validação para garantir que o output tem o formato esperado.
    *   **Entrega:** O primeiro modelo especialista treinado (ex: `crypto_bayesian_moe_specialist_v2.ckpt`). **Este é o primeiro grande marco do projeto.**

---

#### **Fase 3: Construção do Motor de Inferência e Sinais (Semanas 7-9)**

**Objetivo:** Construir o pipeline que consome o modelo treinado e o transforma em sinais de trading acionáveis e ricos em informação, prontos para a tomada de decisão.

*   **Semana 7: Implementação do Motor de Inferência em Batch (SOTA)**
    *   **Atividade Principal:** Desenvolver o motor que carrega o modelo e realiza previsões em batch de forma otimizada.
    *   **Arquivo(s) Chave:** `src/crypto_inference/bayesian_inference_engine.py`.
    *   **Passos:**
        1.  Implementar a classe `BayesianCryptoInferenceEngine`.
        2.  Implementar o método de carregamento do modelo, incluindo a compilação com `torch.compile`.
        3.  Implementar a função principal `predict_batch_with_uncertainty`, que recebe o batch de tensores `[n_assets, 2048, n_features]` e executa a amostragem Monte Carlo para todos os ativos em paralelo.
        4.  Garantir que a saída seja um dicionário de objetos `BayesianPredictionOutput`, um para cada ativo.
    *   **Entrega:** Uma classe robusta que transforma dados de mercado em previsões probabilísticas estruturadas.

*   **Semanas 8-9: Pós-processamento e Geração de Sinais Avançados**
    *   **Atividade Principal:** Traduzir as previsões probabilísticas em métricas de trading e gestão de risco.
    *   **Arquivo(s) Chave:** `src/crypto_inference/bayesian_postprocessor.py`.
    *   **Passos:**
        1.  Implementar a classe `BayesianPostprocessor`.
        2.  Implementar cada uma das funções de cálculo de métricas avançadas, uma por uma: `_compute_directional_probability`, `_compute_bayesian_kelly`, `_detect_volatility_regime`, `_compute_position_size`, etc.
        3.  Integrar a saída do `BayesianCryptoInferenceEngine` com o `BayesianPostprocessor` no notebook `notebooks/inference_demo.ipynb` para visualizar um sinal de trading completo.
    *   **Entrega:** O pipeline de lógica completo, capaz de ir de dados brutos de mercado a um JSON estruturado com sinais de trading acionáveis.

---

#### **Fase 4: Produção, Validação Contínua e Deploy (Semanas 10-12)**

**Objetivo:** Colocar o sistema para operar em tempo real, monitorar sua performance e garantir sua confiabilidade e calibração ao longo do tempo.

*   **Semana 10: Orquestração do Pipeline Real-Time e Deploy**
    *   **Atividade Principal:** Criar o script que une todas as partes e preparar o ambiente de produção.
    *   **Arquivo(s) Chave:** `scripts/real_time_pipeline.py`, `Dockerfile`, `requirements.txt`.
    *   **Passos:**
        1.  Implementar o script `real_time_pipeline.py`, que a cada minuto:
            a. Chama o `BinanceDataCollector` para obter o batch de dados.
            b. Prepara o batch de inferência.
            c. Passa o batch para o `BayesianCryptoInferenceEngine`.
            d. Passa os resultados para o `BayesianPostprocessor`.
            e. Loga ou exibe os sinais de trading finais.
        2.  Configurar o ambiente de produção (AWS g4dn.xlarge), instalar dependências, drivers NVIDIA e TensorRT.
        3.  Fazer o deploy do pipeline.
    *   **Entrega:** Um sistema em operação, gerando previsões e sinais a cada minuto.

*   **Semanas 11-12: Calibração, Monitoramento e Análise de Performance**
    *   **Atividade Principal:** Implementar o feedback loop para avaliar a qualidade e a calibração das previsões em tempo real.
    *   **Arquivo(s) Chave:** `src/crypto_inference/calibration_monitor.py`, `notebooks/calibration_analysis.ipynb`.
    *   **Passos:**
        1.  Implementar a classe `CalibrationMonitor` e o método `update_calibration`.
        2.  Integrar o monitor ao pipeline real-time, salvando predições, incertezas e resultados reais em um banco de dados ou arquivo de log.
        3.  Desenvolver o notebook `calibration_analysis.ipynb` para analisar as métricas de calibração (ECE, PIT, Coverage) e a performance do trading (Sharpe, Drawdown) com base nos dados coletados.
    *   **Entrega:** Um sistema de produção com um framework de monitoramento robusto, permitindo a validação contínua e a tomada de decisões sobre o re-treinamento do modelo.