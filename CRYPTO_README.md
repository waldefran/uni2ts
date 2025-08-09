# 🚀 Crypto Fine-tuning SOTA - Implementação BOOST.MD

Esta implementação aplica **TODAS** as melhorias do BOOST.MD para trading de criptomoedas com Moirai-MoE + Cabeça Bayesiana.

## ✅ **Melhorias BOOST.MD Implementadas**

| Componente | Upgrade Aplicado | Status |
|------------|------------------|--------|
| **Dataset** | Unificado e anônimo (remove `item_id`) | ✅ |
| **Arquitetura** | Moirai-MoE para dataset diverso | ✅ |
| **Features** | Cíclicas explícitas (sin/cos) | ✅ |
| **Normalização** | Por janela: `(valor[t]/valor[t=0]) - 1` | ✅ |
| **Inferência** | Batch paralelo para todos os ativos | ✅ |
| **Bayesiano** | BayesianHead + ELBO mantidos SOTA | ✅ |
| **Monitoramento** | Callbacks SOTA + Métricas Bayesianas | ✅ |
| **CLI Avançado** | Interface completa para pipeline | ✅ |
| **Validação** | Scripts de teste automático | ✅ |

## 🏗️ **Estrutura Criada**

```
├── binance_data/                          # Dados baixados pelo binanceDataloader.py
├── src/uni2ts/
│   ├── model/crypto/
│   │   ├── __init__.py
│   │   └── bayesian_head.py               # ⭐ Cabeça Bayesiana SOTA
│   ├── loss/
│   │   └── bayesian_elbo.py               # ⭐ Loss ELBO Bayesiana
│   ├── data/builder/
│   │   └── crypto.py                      # ⭐ Dataset Builder unificado
│   ├── callbacks/
│   │   ├── __init__.py                    # ⭐ Exposição de callbacks SOTA
│   │   └── bayesian_uncertainty.py       # ⭐ Monitor de incerteza Bayesiana
│   ├── eval_util/
│   │   └── bayesian_metrics.py           # ⭐ Métricas Bayesianas completas
│   └── cli/
│       └── crypto_bayesian.py            # ⭐ CLI avançado para pipeline
├── configs/crypto/
│   └── finetune_bayesian_moe.yaml         # ⭐ Config completa MoE
├── scripts/crypto/
│   ├── prepare_dataset.py                 # Preparação de dados
│   ├── finetune_model.py                  # Fine-tuning script
│   └── validate_sota_pipeline.py          # ⭐ Validação automática SOTA
├── notebooks/crypto/
│   └── data_exploration.ipynb             # Análise exploratória
└── requirements_crypto.txt                # Dependências SOTA
```

## 🎯 **Quick Start**

### **Opção 1: CLI Avançado (Recomendado)** ⭐
```bash
# Validar pipeline completo primeiro
python scripts/crypto/validate_sota_pipeline.py

# CLI completo para fine-tuning
python -m uni2ts.cli.crypto_bayesian \
    --config configs/crypto/finetune_bayesian_moe.yaml \
    --data_path ./binance_data \
    --output_dir ./results
```

### **Opção 2: Kaggle GPU P100 Tutorial** 🎯
Para execução no Kaggle com GPU P100, siga o tutorial completo:
**[KAGGLE_FINETUNING_TUTORIAL.md](KAGGLE_FINETUNING_TUTORIAL.md)**

Inclui:
- ✅ Configuração otimizada para P100 (16GB VRAM)
- ✅ Download automático de dados crypto
- ✅ Pipeline completo em notebook Kaggle
- ✅ Troubleshooting e otimizações
- ✅ Análise e download de resultados

### **Opção 2: Scripts Individuais**

#### 1. **Baixar Dados** (já funcional)
```bash
# O binanceDataloader.py já está baixando os dados
python binanceDataloader.py
```

#### 2. **Instalar Dependências**
```bash
pip install -r requirements_crypto.txt
```

#### 3. **Preparar Dataset**
```bash
python scripts/crypto/prepare_dataset.py \
    --data_path ./binance_data \
    --output_path ./crypto_dataset \
    --assets BTCUSDT ETHUSDT ETHBTC BNBUSDT
```

#### 4. **Fine-tuning**
```bash
python scripts/crypto/finetune_model.py \
    --config configs/crypto/finetune_bayesian_moe.yaml
```

#### 5. **Explorar Dados**
```bash
jupyter lab notebooks/crypto/data_exploration.ipynb
```

## 🧠 **Componentes SOTA**

### **BayesianPredictionHead** ⭐
- ✅ **Incerteza epistêmica** (MC Dropout + pesos variacionais)
- ✅ **Incerteza aleatória** (distribuições paramétricas)  
- ✅ **Student-T distribution** (heavy tails para crypto)
- ✅ **Attention temporal** (interpretabilidade)
- ✅ **Decomposição rigorosa** de incerteza

### **BayesianELBOLoss** ⭐
- ✅ **Evidence Lower Bound** matematicamente rigorosa
- ✅ **Regularização KL** essencial para variacionais
- ✅ **KL annealing** para estabilidade
- ✅ **Student-T likelihood** para heavy tails

### **CryptoDatasetBuilder** ⭐
- ✅ **Dataset unificado** (todos os ativos juntos)
- ✅ **Anonimização** (remove `item_id` no treino)
- ✅ **Normalização por janela** `(valor[t] / valor[t=0]) - 1`
- ✅ **Features cíclicas** sin/cos para tempo
- ✅ **Compatível Moirai-MoE**

### **BayesianUncertaintyMonitor** ⭐ (NOVO)
- ✅ **Monitoramento em tempo real** da incerteza
- ✅ **Decomposição epistêmica/aleatória** por época
- ✅ **Logging para TensorBoard/Wandb**
- ✅ **Alertas automáticos** para overfitting
- ✅ **Histogramas de distribuições**

### **Métricas Bayesianas Completas** ⭐ (NOVO)
- ✅ **BayesianCalibration**: ECE, MCE, reliability diagrams
- ✅ **UncertaintyDecomposition**: Epistêmica vs aleatória
- ✅ **PredictionSharpness**: Concentração de probabilidade
- ✅ **PredictionIntervalCoverage**: Cobertura de intervalos
- ✅ **BayesianMetricsCalculator**: Aggregação automática

### **CLI Crypto Bayesiano** ⭐ (NOVO)
- ✅ **Interface unificada** para todo o pipeline
- ✅ **Validação automática** de configurações
- ✅ **Logging avançado** com níveis configuráveis
- ✅ **Checkpoint management** automático
- ✅ **Resumo de treinamento** com métricas SOTA

## 📊 **Configuração SOTA**

O arquivo `configs/crypto/finetune_bayesian_moe.yaml` implementa:

```yaml
# Moirai-MoE para dataset unificado
model:
  pretrained_model_name_or_path: "Salesforce/moirai-moe-1.0-R-base"
  
  # Cabeça Bayesiana SOTA
  prediction_head:
    _target_: uni2ts.model.crypto.bayesian_head.BayesianPredictionHead
    use_variational_weights: true  # CONFIRMADO
    use_temporal_attention: true
    student_t_df: 4.0              # Heavy tails

# Loss Bayesiana SOTA  
loss_func:
  _target_: uni2ts.loss.bayesian_elbo.BayesianELBOLoss  # CONFIRMADO

# Dataset unificado e anônimo
data:
  _target_: uni2ts.data.builder.crypto.CryptoDatasetBuilder
  config:
    unified_dataset: true          # UPGRADE
    anonymous_training: true       # UPGRADE  
    window_normalization: true     # UPGRADE
    cyclical_features: true        # UPGRADE

# Callbacks SOTA para monitoramento
callbacks:
  - _target_: uni2ts.callbacks.bayesian_uncertainty.BayesianUncertaintyMonitor
    log_interval: 100
    plot_distributions: true
    save_uncertainty_plots: true
  
  - _target_: lightning.pytorch.callbacks.ModelCheckpoint
    monitor: val_loss
    save_top_k: 3
    save_last: true
```

## 🎯 **Features Implementadas**

### **1. Dataset Unificado e Anônimo**
- Todos os ativos (BTC, ETH, etc.) em um dataset único
- Remove `item_id` durante treinamento
- Força aprendizado de padrões universais

### **2. Features Cíclicas Explícitas** 
- `minute_sin/cos`: Ciclo de 60 minutos
- `hour_sin/cos`: Ciclo de 24 horas  
- `weekday_sin/cos`: Ciclo de 7 dias
- Contexto temporal explícito para regimes de mercado

### **3. Normalização por Janela**
- Formula: `(valor[t] / valor[t=0]) - 1`
- Foca na forma do padrão, não escala absoluta
- Melhora convergência Bayesiana

### **4. Processamento Batch Paralelo**
- Todos os ativos processados simultaneamente
- Tensor: `[n_assets, 2048, n_features]`
- Máxima eficiência GPU

## 🔬 **Validação Científica**

### **Incerteza Bayesiana**
- **Epistêmica**: Reduzível com mais dados
- **Aleatória**: Inerente ao mercado crypto
- **Decomposição**: `total = epistêmica + aleatória`

### **Calibração Automática** ⭐ (NOVO)
- **ECE (Expected Calibration Error)**: Mede sobre/sub-confiança
- **MCE (Maximum Calibration Error)**: Worst-case calibration
- **Reliability Diagrams**: Visualização de calibração
- **PIT (Probability Integral Transform)**: Teste de uniformidade

### **Métricas de Cobertura** ⭐ (NOVO)
- **Prediction Interval Coverage**: 50%, 80%, 95% intervalos
- **Conditional Coverage**: Cobertura por volatilidade
- **Width-Coverage Trade-off**: Eficiência dos intervalos

### **Análise de Sharpness** ⭐ (NOVO)
- **Entropy das Predições**: Concentração de probabilidade
- **Variance Decomposition**: Contribuições das fontes
- **Confidence Histograms**: Distribuição de confiança

### **Métricas Trading**
- **Sharpe Bayesiano**: Ajustado por incerteza
- **Kelly Criterion**: Posicionamento ótimo
- **Regime Detection**: Bull/bear markets

## 🚀 **Próximos Passos**

1. ✅ **Dados preparados** (binanceDataloader.py)
2. ✅ **Estrutura criada** (todos os arquivos SOTA)
3. ✅ **Callbacks implementados** (monitoramento Bayesiano)
4. ✅ **Métricas SOTA** (calibração + cobertura)
5. ✅ **CLI avançado** (interface unificada)
6. ✅ **Validação automática** (script de teste)
7. 🔄 **Fine-tuning** (executar pipeline)
8. 🔄 **Validação final** (métricas Bayesianas)
9. 🔄 **Inferência** (pipeline tempo real)

## 🧪 **Validação Automática** ⭐ (NOVO)

Execute o script de validação para verificar toda a implementação:

```bash
python scripts/crypto/validate_sota_pipeline.py
```

**Verificações incluídas:**
- ✅ Importação de todos os componentes SOTA
- ✅ Configuração YAML válida
- ✅ Callbacks funcionais
- ✅ Métricas Bayesianas operacionais
- ✅ CLI responsivo
- ✅ Compatibilidade com PyTorch Lightning
- ✅ Validação de tipos e estruturas

## 📝 **Confirmações BOOST.MD**

> ✅ **BayesianPredictionHead mantida completa** com MC Dropout e pesos variacionais  
> ✅ **BayesianELBOLoss mantida completa** com regularização KL  
> ✅ **Nenhum downgrade** - Estado da arte preservado  
> ✅ **Todas as melhorias** do BOOST.MD implementadas  
> ✅ **Callbacks SOTA** para monitoramento em tempo real  
> ✅ **Métricas Bayesianas** científicamente rigorosas  
> ✅ **CLI avançado** para pipeline completo  
> ✅ **Validação automática** de toda implementação  

## 🎉 **Estado da Arte Completo**

Esta implementação representa o **verdadeiro estado da arte** para trading quantitativo com deep learning, combinando:

- 🧠 **Foundation Model** (Moirai-MoE)
- 📊 **Quantificação de Incerteza** (Bayesiana)
- 🔄 **Dataset Otimizado** (unificado + anônimo)
- ⚡ **Performance Máxima** (batch paralelo)
- 🎯 **Trading Inteligente** (gestão de risco probabilística)
- 📈 **Monitoramento SOTA** (callbacks + métricas)
- 🛠️ **Tooling Avançado** (CLI + validação)

---

## 🏆 **Ready for Fine-tuning!**

Todos os componentes SOTA estão implementados e prontos para execução. 🚀

### **Como começar:**
```bash
# 1. Validar implementação completa
python scripts/crypto/validate_sota_pipeline.py

# 2. Executar fine-tuning com CLI avançado
python -m uni2ts.cli.crypto_bayesian \
    --config configs/crypto/finetune_bayesian_moe.yaml \
    --data_path ./binance_data

# 3. Monitorar métricas Bayesianas em tempo real
tensorboard --logdir ./results/logs
```
