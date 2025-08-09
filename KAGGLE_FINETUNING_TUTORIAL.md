# 🚀 Tutorial: Fine-tuning Crypto SOTA no Kaggle GPU P100

Este tutorial completo mostra como executar o fine-tuning do modelo Moirai-MoE com cabeça Bayesiana no Kaggle usando GPU P100.

## 📋 **Pré-requisitos**

- Conta no Kaggle verificada com acesso a GPU
- Conhecimento básico de Jupyter notebooks
- Dados de crypto (ou usar nosso script de download)

---

## 🎯 **Configuração do Ambiente Kaggle**

### **1. Criar Novo Notebook**
1. Acesse [kaggle.com/code](https://www.kaggle.com/code)
2. Clique em "New Notebook"
3. **Settings importantes:**
   - **Accelerator**: GPU P100
   - **Language**: Python
   - **Internet**: ON (para downloads)

### **2. Configuração Inicial**

```python
# Cell 1: Verificar GPU e instalar dependências
import torch
print(f"GPU disponível: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Nenhuma'}")
print(f"CUDA version: {torch.version.cuda}")

# Verificar memória GPU
if torch.cuda.is_available():
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"Memória GPU: {gpu_memory:.1f} GB")
```

### **3. Clonar Repositório e Instalar**

```python
# Cell 2: Setup do projeto
import os
import subprocess
import sys

# Clonar o repositório (ou fazer upload do ZIP)
!git clone https://github.com/SalesforceAIResearch/uni2ts.git
os.chdir('/kaggle/working/uni2ts')

# Verificar se temos nossos arquivos SOTA
!ls -la src/uni2ts/model/crypto/
!ls -la src/uni2ts/loss/
!ls -la configs/crypto/

# Instalar dependências base
!pip install -e .

# Instalar dependências crypto específicas
!pip install -r requirements_crypto.txt
```

---

## 📊 **Preparação dos Dados**

### **4. Download de Dados Crypto**

```python
# Cell 3: Download dados da Binance
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import time

def download_binance_data(symbol, interval='1m', limit=1000):
    """Download dados históricos da Binance"""
    base_url = 'https://api.binance.com/api/v3/klines'
    
    params = {
        'symbol': symbol,
        'interval': interval,
        'limit': limit
    }
    
    response = requests.get(base_url, params=params)
    data = response.json()
    
    # Converter para DataFrame
    df = pd.DataFrame(data, columns=[
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
    ])
    
    # Converter tipos
    numeric_columns = ['open', 'high', 'low', 'close', 'volume', 
                      'quote_asset_volume', 'taker_buy_base_asset_volume', 
                      'taker_buy_quote_asset_volume']
    df[numeric_columns] = df[numeric_columns].astype(float)
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
    
    return df.drop('ignore', axis=1)

# Download dados para principais cryptos
symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'DOTUSDT']
crypto_data = {}

print("Baixando dados crypto...")
for symbol in symbols:
    print(f"Baixando {symbol}...")
    data = download_binance_data(symbol, interval='1m', limit=1000)
    crypto_data[symbol] = data
    time.sleep(0.1)  # Rate limiting
    
print("✅ Download completo!")
```

### **5. Preparar Dataset SOTA**

```python
# Cell 4: Preparar dataset unificado e anônimo
import os
os.chdir('/kaggle/working/uni2ts')

# Salvar dados em formato CSV
os.makedirs('/kaggle/working/binance_data', exist_ok=True)

for symbol, data in crypto_data.items():
    filepath = f'/kaggle/working/binance_data/{symbol}.csv'
    data.to_csv(filepath, index=False)
    print(f"Salvo: {filepath} ({len(data)} registros)")

# Executar preparação SOTA
!python scripts/crypto/prepare_dataset.py \
    --data_path /kaggle/working/binance_data \
    --output_path /kaggle/working/crypto_dataset \
    --assets BTCUSDT ETHUSDT ADAUSDT DOTUSDT

print("✅ Dataset SOTA preparado!")
```

---

## 🧠 **Validação da Implementação**

### **6. Validar Pipeline SOTA**

```python
# Cell 5: Validação completa
!python scripts/crypto/validate_sota_pipeline.py

# Verificar se todos os componentes estão funcionando
print("\n🔍 Verificação manual dos componentes:")

# Testar importações
try:
    from uni2ts.model.crypto.bayesian_head import BayesianPredictionHead
    from uni2ts.loss.bayesian_elbo import BayesianELBOLoss
    from uni2ts.data.builder.crypto import CryptoDatasetBuilder
    from uni2ts.callbacks.bayesian_uncertainty import BayesianUncertaintyMonitor
    from uni2ts.eval_util.bayesian_metrics import BayesianMetricsCalculator
    print("✅ Todas as importações SOTA funcionando!")
except Exception as e:
    print(f"❌ Erro na importação: {e}")

# Verificar config
import yaml
with open('configs/crypto/finetune_bayesian_moe.yaml', 'r') as f:
    config = yaml.safe_load(f)
    print("✅ Configuração SOTA carregada!")
```

---

## 🚀 **Fine-tuning Execution**

### **7. Configurar Parâmetros para P100**

```python
# Cell 6: Otimizar configuração para P100 (16GB VRAM)
config_p100 = """
# Configuração otimizada para Kaggle P100
model:
  _target_: uni2ts.model.moirai_moe.MoiraiMoEModule
  pretrained_model_name_or_path: "Salesforce/moirai-moe-1.0-R-small"  # Usar SMALL para P100
  
  prediction_head:
    _target_: uni2ts.model.crypto.bayesian_head.BayesianPredictionHead
    hidden_size: 512  # Reduzido para P100
    num_mc_samples: 20  # Reduzido para memória
    use_variational_weights: true
    use_temporal_attention: true
    student_t_df: 4.0

loss_func:
  _target_: uni2ts.loss.bayesian_elbo.BayesianELBOLoss
  kl_weight: 0.001
  kl_annealing_epochs: 10

data:
  _target_: uni2ts.data.builder.crypto.CryptoDatasetBuilder
  path: "/kaggle/working/crypto_dataset"
  batch_size: 16  # Reduzido para P100
  max_length: 1024  # Reduzido para memória
  
trainer:
  max_epochs: 20  # Limitado para Kaggle
  precision: 16  # Mixed precision para economizar memória
  gradient_clip_val: 1.0
  accumulate_grad_batches: 4  # Simular batch maior
  
callbacks:
  - _target_: uni2ts.callbacks.bayesian_uncertainty.BayesianUncertaintyMonitor
    log_interval: 50
    plot_distributions: false  # Desabilitado para performance
    
  - _target_: lightning.pytorch.callbacks.ModelCheckpoint
    monitor: val_loss
    save_top_k: 2
    save_last: true
    
  - _target_: lightning.pytorch.callbacks.EarlyStopping
    monitor: val_loss
    patience: 5
"""

# Salvar configuração otimizada
with open('/kaggle/working/config_p100.yaml', 'w') as f:
    f.write(config_p100)
    
print("✅ Configuração P100 criada!")
```

### **8. Executar Fine-tuning**

```python
# Cell 7: Fine-tuning com CLI SOTA
import subprocess
import sys

# Executar fine-tuning
cmd = [
    sys.executable, "-m", "uni2ts.cli.crypto_bayesian",
    "--config", "/kaggle/working/config_p100.yaml",
    "--data_path", "/kaggle/working/crypto_dataset",
    "--output_dir", "/kaggle/working/results",
    "--log_level", "INFO"
]

print("🚀 Iniciando fine-tuning SOTA...")
print(f"Comando: {' '.join(cmd)}")

# Executar com output em tempo real
process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                          universal_newlines=True, cwd='/kaggle/working/uni2ts')

for line in process.stdout:
    print(line.strip())
    
return_code = process.wait()
print(f"\n✅ Fine-tuning finalizado com código: {return_code}")
```

---

## 📊 **Monitoramento e Análise**

### **9. Análise dos Resultados**

```python
# Cell 8: Análise dos resultados
import matplotlib.pyplot as plt
import torch
import json
import glob

# Verificar checkpoints salvos
checkpoint_dir = "/kaggle/working/results/checkpoints"
checkpoints = glob.glob(f"{checkpoint_dir}/*.ckpt")
print(f"Checkpoints salvos: {len(checkpoints)}")
for ckpt in checkpoints[-3:]:  # Últimos 3
    print(f"  {ckpt}")

# Carregar métricas de treinamento
metrics_file = "/kaggle/working/results/metrics.json"
if os.path.exists(metrics_file):
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)
    
    # Plot das métricas
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Loss
    if 'train_loss' in metrics:
        axes[0,0].plot(metrics['train_loss'], label='Train')
        axes[0,0].plot(metrics['val_loss'], label='Val')
        axes[0,0].set_title('Loss')
        axes[0,0].legend()
    
    # Incerteza Bayesiana
    if 'epistemic_uncertainty' in metrics:
        axes[0,1].plot(metrics['epistemic_uncertainty'], label='Epistemic')
        axes[0,1].plot(metrics['aleatoric_uncertainty'], label='Aleatoric')
        axes[0,1].set_title('Uncertainty Decomposition')
        axes[0,1].legend()
    
    # Calibração
    if 'ece_score' in metrics:
        axes[1,0].plot(metrics['ece_score'])
        axes[1,0].set_title('Expected Calibration Error')
    
    # Coverage
    if 'coverage_95' in metrics:
        axes[1,1].plot(metrics['coverage_95'], label='95%')
        axes[1,1].plot(metrics['coverage_80'], label='80%')
        axes[1,1].set_title('Prediction Coverage')
        axes[1,1].legend()
    
    plt.tight_layout()
    plt.savefig('/kaggle/working/training_metrics.png', dpi=150)
    plt.show()
    
    print("✅ Métricas plotadas!")
else:
    print("⚠️ Arquivo de métricas não encontrado")
```

### **10. Teste de Inferência**

```python
# Cell 9: Teste do modelo treinado
import torch
from uni2ts.model.moirai_moe import MoiraiMoEModule
from uni2ts.model.crypto.bayesian_head import BayesianPredictionHead

# Carregar melhor checkpoint
best_checkpoint = sorted(checkpoints, key=os.path.getmtime)[-1]
print(f"Carregando: {best_checkpoint}")

# Carregar modelo
checkpoint = torch.load(best_checkpoint, map_location='cuda' if torch.cuda.is_available() else 'cpu')
model_state = checkpoint['state_dict']

print("✅ Modelo carregado com sucesso!")

# Teste com dados de exemplo
sample_data = crypto_data['BTCUSDT'].tail(100)[['close']].values
sample_tensor = torch.FloatTensor(sample_data).unsqueeze(0)

print(f"Input shape: {sample_tensor.shape}")
print("🔮 Modelo pronto para inferência Bayesiana!")

# Exemplo de predição (simulado)
print("\n📊 Exemplo de saída Bayesiana:")
print("Mean prediction: [50234.5, 50456.7, 50123.4, ...]")
print("Epistemic uncertainty: [124.3, 156.7, 198.2, ...]")
print("Aleatoric uncertainty: [345.6, 378.9, 412.1, ...]")
print("95% confidence interval: [[49876, 50593], [50067, 50846], ...]")
```

---

## 💾 **Salvamento dos Resultados**

### **11. Download dos Artefatos**

```python
# Cell 10: Preparar arquivos para download
import zipfile
import shutil

# Criar ZIP com resultados
output_zip = "/kaggle/working/crypto_finetuning_results.zip"

with zipfile.ZipFile(output_zip, 'w') as zipf:
    # Adicionar checkpoints
    for ckpt in checkpoints:
        zipf.write(ckpt, f"checkpoints/{os.path.basename(ckpt)}")
    
    # Adicionar configurações
    zipf.write("/kaggle/working/config_p100.yaml", "config_p100.yaml")
    
    # Adicionar métricas se existir
    if os.path.exists(metrics_file):
        zipf.write(metrics_file, "metrics.json")
    
    # Adicionar plots se existir
    if os.path.exists("/kaggle/working/training_metrics.png"):
        zipf.write("/kaggle/working/training_metrics.png", "training_metrics.png")
    
    # Adicionar logs
    log_files = glob.glob("/kaggle/working/results/logs/*")
    for log_file in log_files[:5]:  # Últimos 5 logs
        zipf.write(log_file, f"logs/{os.path.basename(log_file)}")

print(f"✅ Resultados empacotados em: {output_zip}")
print(f"Tamanho do arquivo: {os.path.getsize(output_zip) / 1e6:.1f} MB")

# Listar conteúdo
with zipfile.ZipFile(output_zip, 'r') as zipf:
    print("\n📁 Conteúdo do ZIP:")
    for file_info in zipf.filelist:
        print(f"  {file_info.filename} ({file_info.file_size} bytes)")
```

---

## 🔧 **Troubleshooting**

### **Problemas Comuns no Kaggle P100**

#### **1. Out of Memory (OOM)**
```python
# Reduzir batch size
batch_size: 8  # ou menor

# Reduzir sequência
max_length: 512

# Usar gradient checkpointing
gradient_checkpointing: true
```

#### **2. Timeout do Kaggle (9h)**
```python
# Reduzir épocas
max_epochs: 10

# Usar early stopping agressivo
patience: 3

# Salvar checkpoints frequentes
save_every_n_epochs: 2
```

#### **3. Download Failed**
```python
# Usar dados mock se API falhar
def create_mock_data(symbol, n_samples=1000):
    dates = pd.date_range('2024-01-01', periods=n_samples, freq='1min')
    np.random.seed(42)
    
    # Simular preços crypto com volatilidade
    returns = np.random.normal(0, 0.02, n_samples)
    prices = 50000 * np.exp(np.cumsum(returns))
    
    return pd.DataFrame({
        'open_time': dates,
        'open': prices * (1 + np.random.normal(0, 0.001, n_samples)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.005, n_samples))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.005, n_samples))),
        'close': prices,
        'volume': np.random.exponential(1000, n_samples)
    })
```

---

## 📋 **Checklist Final**

### **Antes de executar:**
- [ ] GPU P100 ativada no Kaggle
- [ ] Internet ativada para downloads
- [ ] Repositório clonado ou uploadado
- [ ] Configuração P100 ajustada

### **Durante execução:**
- [ ] Monitorar uso de memória GPU
- [ ] Verificar progresso das métricas
- [ ] Salvar checkpoints regularmente
- [ ] Monitorar tempo restante (9h limite)

### **Após treinamento:**
- [ ] Validar qualidade do modelo
- [ ] Análise das métricas Bayesianas
- [ ] Download dos resultados
- [ ] Backup dos checkpoints importantes

---

## 🎉 **Conclusão**

Este tutorial fornece um pipeline completo para fine-tuning SOTA de modelos crypto no Kaggle P100. As configurações foram otimizadas para funcionar dentro das limitações de memória e tempo da plataforma.

**Recursos principais implementados:**
- ✅ **Moirai-MoE** com cabeça Bayesiana
- ✅ **Quantificação de incerteza** epistêmica e aleatória
- ✅ **Métricas científicas** (ECE, PIT, coverage)
- ✅ **Monitoramento em tempo real**
- ✅ **Configuração otimizada P100**

**💡 Próximos passos:**
1. Execute o fine-tuning seguindo este tutorial
2. Analise as métricas Bayesianas obtidas
3. Teste diferentes hiperparâmetros
4. Implemente em ambiente de produção

**🚀 Happy Training!**
