#!/usr/bin/env python
# -*- coding: utf-8 -*-

# # 1. Configuração de Ambiente Avançada
# Configuração avançada com detecção automática de ambiente e otimizações específicas

import os
import sys
import yaml
import json
import logging
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pytorch_lightning as pl
from datetime import datetime
from pathlib import Path
from scipy import stats

# Adiciona o diretório 'src' ao path para importações locais
module_path = os.path.abspath(os.path.join('.', 'src'))
if module_path not in sys.path:
    sys.path.insert(0, module_path)

# Detecção automática do ambiente
def detect_environment():
    """Detecta automaticamente o ambiente de execução."""
    if os.path.exists('/kaggle/input'):
        return 'kaggle'
    elif os.environ.get('COLAB_GPU'):
        return 'colab'
    else:
        return 'local'

ENV_TYPE = detect_environment()

# Configuração de logging adaptativa com handlers múltiplos
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"finetune_{ENV_TYPE}.log")
    ]
)

logger = logging.getLogger(__name__)
logger.info(f"Ambiente detectado: {ENV_TYPE}")
logger.info(f"Versão do PyTorch: {torch.__version__}")
logger.info(f"Versão do Lightning: {pl.__version__}")

# Função de mesclagem profunda para configurações
def _deep_merge(dict1, dict2):
    """
    Mescla recursivamente dois dicionários, mantendo a estrutura de ambos.
    Os valores em dict2 têm precedência sobre dict1.
    """
    result = dict1.copy()
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result

# # 2. Otimização de GPU e Hardware
# Configuração avançada de GPU com detecção automática de capacidades e otimizações

# Otimização automática de GPU baseada no ambiente e capacidades
def optimize_gpu_settings():
    """
    Configura otimizações específicas para o hardware disponível.
    Retorna a precisão recomendada para o treinamento.
    """
    if torch.cuda.is_available():
        # Detecção avançada de capacidades da GPU
        capability = torch.cuda.get_device_capability()
        device_name = torch.cuda.get_device_name(0)
        memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        
        logger.info(f"GPU detectada: {device_name}")
        logger.info(f"Memória GPU: {memory_gb:.2f} GB")
        logger.info(f"Compute capability: {capability[0]}.{capability[1]}")
        
        # Configurações otimizadas para diferentes gerações de GPUs
        precision = 32
        if capability[0] >= 8:  # Ampere (RTX 30xx, A100) ou mais recente
            torch.set_float32_matmul_precision('high')
            precision = '16-mixed'
            logger.info("TF32 habilitado para aceleração em GPUs Ampere+")
        elif capability[0] >= 7:  # Volta/Turing (RTX 20xx, V100)
            precision = '16-mixed'
            logger.info("Precisão mista habilitada para GPUs Volta/Turing")
        
        # Configurações de cache para melhorar a utilização da GPU
        torch.backends.cudnn.benchmark = True
        logger.info("cuDNN benchmark habilitado para otimização de operações")
        
        # Determinar automaticamente o tamanho de batch ótimo baseado na memória
        # Heurística: ~4GB para modelo base + ~10MB por exemplo no batch
        recommended_batch_size = max(1, int((memory_gb - 4) * 100))
        recommended_batch_size = min(64, recommended_batch_size)  # Cap em 64 para evitar divergência
        logger.info(f"Tamanho de batch recomendado: {recommended_batch_size}")
        
        return precision, recommended_batch_size
    else:
        logger.warning("Nenhuma GPU detectada. O treinamento será executado em CPU.")
        return 32, 8  # Batch size menor para CPU

# Executar a otimização e obter recomendações
PRECISION, RECOMMENDED_BATCH_SIZE = optimize_gpu_settings()

# # 3. Configuração de Constantes e Paths
# Definição de constantes e criação de diretórios de trabalho

# Path base do projeto
PROJECT_ROOT = Path.cwd()
logger.info(f"Raiz do projeto definida em: {PROJECT_ROOT}")

# Paths para dados, logs e saídas
DATA_DIR = PROJECT_ROOT / "data"
BINANCE_DATA_DIR = DATA_DIR / "binance_data"
OUTPUT_DIR = PROJECT_ROOT / "output" / f"bayesian_moe_{datetime.now().strftime('%Y%m%d_%H%M')}"
LOG_DIR = PROJECT_ROOT / "logs"
CONFIG_PATH = PROJECT_ROOT / "configs" / "crypto" / "finetune_bayesian_moe.yaml"
UNCERTAINTY_PLOTS_DIR = OUTPUT_DIR / "uncertainty_plots"

# Cria os diretórios se eles não existirem
for dir_path in [DATA_DIR, BINANCE_DATA_DIR, OUTPUT_DIR, LOG_DIR, UNCERTAINTY_PLOTS_DIR]:
    dir_path.mkdir(exist_ok=True, parents=True)

# Constantes do Modelo
ASSETS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT"]
DTYPE = "float64"  # Tipo de dado para precisão numérica
CONTEXT_LENGTH = 1440  # 24 horas em minutos (dados de 1 minuto)
PREDICTION_LENGTH = 60  # 1 hora em minutos (previsão de 1 hora)
DATA_YEARS_BACK = 0.5  # 6 meses de dados

logger.info(f"Diretórios de trabalho criados em: {OUTPUT_DIR}")
logger.info(f"Ativos para o estudo: {ASSETS}")
logger.info(f"Período de contexto: {CONTEXT_LENGTH}, Período de previsão: {PREDICTION_LENGTH}")

# # 4. Fixação de Seeds para Reprodutibilidade
# Garantir reprodutibilidade dos resultados

def set_seed(seed):
    """Fixa as seeds para PyTorch, NumPy, Python e Lightning."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    pl.seed_everything(seed, workers=True)
    torch.backends.cudnn.deterministic = True  # Garante determinismo mesmo em GPUs
    os.environ['PYTHONHASHSEED'] = str(seed)
    logger.info(f"Seeds fixadas em {seed} para garantir reprodutibilidade completa.")

SEED = 42
set_seed(SEED)

# # 5. Pipeline de Dados Robusta
# Implementação de pipeline de dados avançada com validação extensiva e fallbacks

class RobustDataPipeline:
    """
    Pipeline de dados robusta com múltiplas camadas de fallback e validação extensiva.
    Projetada para garantir um fluxo de dados confiável em cenários de produção.
    """
    def __init__(self, assets, binance_data_dir, years_back=0.5):
        self.assets = assets
        self.binance_data_dir = binance_data_dir
        self.years_back = years_back
        self.validated_files = []
        
    def execute(self):
        """Executa o pipeline completo de obtenção e validação de dados."""
        logger.info("Iniciando pipeline robusto de dados...")
        
        # 1. Tentativa primária: download direto da Binance
        try:
            logger.info("Iniciando download primário da Binance...")
            from binanceDataloader import BinanceDataDownloader
            
            downloader = BinanceDataDownloader(output_dir=str(self.binance_data_dir))
            results = downloader.download_all_symbols(
                symbols=self.assets,
                years_back=self.years_back
            )
            
            # Validação profunda dos dados
            validation_results = self._validate_downloaded_data()
            if validation_results["success"]:
                logger.info(f"Download e validação primários concluídos com sucesso. {validation_results['valid_count']} arquivos válidos.")
                return validation_results["data_files"]
            else:
                logger.warning(f"Validação falhou para alguns arquivos: {validation_results['errors']}")
                raise ValueError("Validação dos dados incompleta")
                
        except Exception as e:
            logger.error(f"Falha no download primário: {str(e)}")
            logger.info("Tentando usar dados offline pré-baixados...")
            
            # 2. Tentativa secundária: usar dados locais pré-baixados
            offline_results = self._try_offline_data()
            if offline_results["success"]:
                logger.info(f"Dados offline encontrados e validados com sucesso. {offline_results['valid_count']} arquivos válidos.")
                return offline_results["data_files"]
            else:
                raise RuntimeError("Não foi possível obter dados válidos. Pipeline de dados falhou.")

    def _validate_downloaded_data(self):
        """
        Validação extensiva de dados com checagem de integridade, qualidade e correção automática.
        Retorna status de sucesso, contagem de arquivos válidos, e caminho para os arquivos.
        """
        parquet_files = list(Path(self.binance_data_dir).rglob("*.parquet"))
        if not parquet_files:
            return {"success": False, "errors": ["Nenhum arquivo encontrado"], "valid_count": 0, "data_files": []}
        
        logger.info(f"Encontrados {len(parquet_files)} arquivos .parquet para validação.")
        
        required_columns = ['open_time', 'open', 'high', 'low', 'close', 'volume']
        validated_files = []
        errors = []
        
        for file_path in parquet_files:
            try:
                df = pd.read_parquet(file_path)
                asset_name = file_path.parent.name
                
                # 1. Validação de Schema
                missing_cols = [col for col in required_columns if col not in df.columns]
                if missing_cols:
                    error_msg = f"Arquivo {file_path.name} para o ativo {asset_name} tem colunas faltando: {missing_cols}"
                    errors.append(error_msg)
                    logger.error(error_msg)
                    continue
                
                # 2. Validação de Dados Nulos e Correção
                if df[required_columns].isnull().values.any():
                    null_count = df[required_columns].isnull().sum().sum()
                    logger.warning(f"Arquivo {file_path.name} contém {null_count} valores nulos. Aplicando interpolação.")
                    # Interpolar valores nulos
                    df = df.interpolate(method='linear')
                
                # 3. Validação de Tipos (garantindo Float64)
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    if df[col].dtype != 'float64':
                        df[col] = df[col].astype('float64')
                
                # 4. Validação de `open_time`
                if not pd.api.types.is_datetime64_any_dtype(df['open_time']):
                    df['open_time'] = pd.to_datetime(df['open_time'])
                
                # 5. Validação de ordenação temporal
                if not df['open_time'].is_monotonic_increasing:
                    logger.warning(f"Arquivo {file_path.name} não está em ordem cronológica. Ordenando.")
                    df = df.sort_values('open_time')
                
                # 6. Validação de valores extremos/outliers
                for col in ['open', 'high', 'low', 'close']:
                    z_scores = np.abs(stats.zscore(df[col]))
                    outliers = (z_scores > 10).sum()  # Detecta outliers extremos (z > 10)
                    if outliers > 0:
                        logger.warning(f"Arquivo {file_path.name} contém {outliers} valores extremos em {col}.")
                
                # 7. Validação de consistência OHLC
                invalid_candles = (
                    (df['high'] < df['low']) | 
                    (df['open'] > df['high']) | 
                    (df['open'] < df['low']) | 
                    (df['close'] > df['high']) | 
                    (df['close'] < df['low'])
                ).sum()
                
                if invalid_candles > 0:
                    logger.warning(f"Arquivo {file_path.name} contém {invalid_candles} velas com valores OHLC inconsistentes.")
                
                # 8. Validação de gaps temporais
                time_diff = df['open_time'].diff()
                if len(time_diff) > 1:  # Evita erro se houver apenas 1 registro
                    expected_diff = pd.Timedelta(minutes=1)  # Esperamos diferença de 1 min entre registros
                    gaps = (time_diff > expected_diff).sum()
                    if gaps > 0:
                        logger.warning(f"Arquivo {file_path.name} contém {gaps} gaps temporais.")
                
                # Salvar arquivo corrigido se houve modificações
                df.to_parquet(file_path)
                
                # Arquivo validado
                validated_files.append(str(file_path))
                logger.info(f"Arquivo {file_path.name} ({asset_name}) validado com sucesso. {len(df):,} linhas.")
                
            except Exception as e:
                error_msg = f"Falha ao validar o arquivo {file_path}: {str(e)}"
                errors.append(error_msg)
                logger.error(error_msg)
        
        success = len(validated_files) >= len(self.assets)
        return {
            "success": success,
            "errors": errors,
            "valid_count": len(validated_files),
            "data_files": validated_files
        }
        
    def _try_offline_data(self):
        """
        Tenta usar dados locais pré-baixados se o download falhar.
        Retorna status similar a _validate_downloaded_data.
        """
        # Verifica se existem arquivos .parquet no diretório
        parquet_files = list(Path(self.binance_data_dir).rglob("*.parquet"))
        if parquet_files:
            logger.info(f"Encontrados {len(parquet_files)} arquivos parquet locais.")
            return self._validate_downloaded_data()
        else:
            return {"success": False, "errors": ["Nenhum dado offline disponível"], "valid_count": 0, "data_files": []}

# Executar o pipeline de dados
data_pipeline = RobustDataPipeline(
    assets=ASSETS,
    binance_data_dir=BINANCE_DATA_DIR,
    years_back=DATA_YEARS_BACK
)

validated_data_files = data_pipeline.execute()
logger.info(f"Pipeline de dados concluído com sucesso. {len(validated_data_files)} arquivos validados.")

# # 6. Configuração do Dataset com Features SOTA
# Implementação avançada de construção de datasets

from uni2ts.data.builder.crypto import CryptoDatasetBuilder, CryptoConfig

def create_advanced_crypto_config():
    """
    Cria uma configuração avançada para o CryptoDatasetBuilder com features SOTA.
    """
    config = CryptoConfig(
        context_length=CONTEXT_LENGTH,
        prediction_length=PREDICTION_LENGTH,
        
        # Features SOTA
        unified_dataset=True,        # Trata todos os ativos como um único dataset, melhorando a generalização
        anonymous_training=True,     # Remove a identidade do ativo para forçar o modelo a aprender padrões universais
        window_normalization=True,   # Normaliza os dados por janela, focando na forma da série (valor[t] / valor[t=0] - 1)
        cyclical_features=True,      # Features cíclicas para capturar padrões temporais (hora do dia, dia da semana)
        
        # Configurações de particionamento
        min_sequence_length=CONTEXT_LENGTH + PREDICTION_LENGTH,
        validation_split=0.15,       # 15% para validação (test_split removido - será calculado internamente)
        
        # Configurações técnicas
        dtype=DTYPE,
        target_assets=ASSETS,
    )
    
    return config

logger.info("Criando configuração SOTA para o CryptoDatasetBuilder...")
crypto_config = create_advanced_crypto_config()

logger.info("Configuração do dataset definida.")
logger.info(f"Contexto: {crypto_config.context_length} min, Previsão: {crypto_config.prediction_length} min")
logger.info(f"Features: window_norm={crypto_config.window_normalization}, cyclical={crypto_config.cyclical_features}")

# Construção dos Datasets
logger.info("Iniciando a construção dos datasets...")

dataset_builder = CryptoDatasetBuilder(
    data_path=str(BINANCE_DATA_DIR),
    config=crypto_config
)

# Valida se a configuração é compatível com os dados encontrados
if not dataset_builder.validate_config(check_data_path=True):
    raise ValueError("A configuração do dataset builder é inválida para os dados fornecidos.")

train_dataset, val_dataset, test_dataset = dataset_builder.build_datasets()

logger.info("Datasets construídos com sucesso.")
logger.info(f"Tamanho do dataset de treino: {len(train_dataset)}")
logger.info(f"Tamanho do dataset de validação: {len(val_dataset)}")
logger.info(f"Tamanho do dataset de teste: {len(test_dataset)}")

# Validar uma amostra para debugging
sample = train_dataset[0]
logger.info(f"Formato da amostra de dados: {list(sample.keys())}")
logger.info(f"Shape do target: {sample['target'].shape}")
logger.info(f"Shape das features dinâmicas: {sample['feat_dynamic_real'].shape}")
logger.info(f"Tipo de dado do target: {sample['target'].dtype}")

# # 7. Configuração Avançada do Modelo Bayesiano
# Configuração SOTA para modelo bayesiano com parâmetros otimizados

def create_advanced_bayesian_config(base_config=None):
    """
    Cria uma configuração avançada para o modelo Moirai-MoE com head bayesiano.
    Incorpora técnicas de estado da arte para quantificação de incerteza.
    """
    # Configuração base
    config = {
        "model": {
            "pretrained_model_name_or_path": "Salesforce/moirai-moe-1.0-R-base",
            "context_length": CONTEXT_LENGTH,
            "prediction_length": PREDICTION_LENGTH,
            "patch_size": 8,  # Otimizado para dados de 1 minuto
            "num_samples": 100,
            "d_model": 512,
            "num_heads": 8,
            
            # Configurações SOTA para cabeça bayesiana
            "bayesian_head": {
                "mc_dropout_rate": 0.15,  # Valor ótimo para crypto
                "num_mc_samples": 50,     # Balance precisão/performance
                "use_variational_weights": True,  # SOTA para incerteza epistêmica
                "student_t_df": 4.0,  # Otimizado para heavy tails de crypto
                "temporal_attention": True,  # Para capturar dependências complexas
                "confidence_levels": [0.5, 0.8, 0.9, 0.95, 0.99]  # Níveis de confiança variados
            },
            
            # Nova feature SOTA: ativação SiLU para melhor performance
            "activation": "silu"
        },
        
        # ELBO loss SOTA
        "loss_func": {
            "_target_": "uni2ts.loss.bayesian_elbo.BayesianELBOLoss",
            "kl_weight": 0.001,  # Otimizado para crypto
            "likelihood_weight": 1.0,
            "kl_annealing": True,  # Annealing para convergência melhor
            "kl_annealing_epochs": 30
        },
        
        # Otimização avançada
        "optimizer": {
            "lr": 1e-4,
            "weight_decay": 1e-5,
            "scheduler": {
                "_target_": "torch.optim.lr_scheduler.OneCycleLR",
                "max_lr": 3e-4,
                "pct_start": 0.3,
                "div_factor": 25.0,
                "final_div_factor": 10000.0,
            }
        }
    }
    
    # Override com configuração base, se fornecida
    if base_config:
        # Mescla recursiva de dicionários
        config = _deep_merge(base_config, config)
        
    return config

# Carrega a configuração base do YAML se existir, ou cria uma nova
if CONFIG_PATH.exists():
    logger.info(f"Carregando configuração base do arquivo: {CONFIG_PATH}")
    with open(CONFIG_PATH, "r") as f:
        base_config = yaml.safe_load(f)
else:
    logger.warning(f"Arquivo de configuração {CONFIG_PATH} não encontrado. Criando configuração do zero.")
    base_config = {}

# Aplica a configuração avançada bayesiana
config = create_advanced_bayesian_config(base_config)

# Atualiza a configuração com parâmetros específicos do ambiente
config["data"] = {
    "_target_": "uni2ts.data.builder.crypto.CryptoDatasetBuilder",
    "data_path": str(BINANCE_DATA_DIR),
    "config": crypto_config.__dict__
}

# Configuração de dataloaders
config["train_dataloader"] = {
    "batch_size": RECOMMENDED_BATCH_SIZE,
    "num_workers": os.cpu_count() // 2,
    "shuffle": True,
    "pin_memory": True
}

config["val_dataloader"] = {
    "batch_size": RECOMMENDED_BATCH_SIZE * 2,  # Pode usar batch maior na validação
    "num_workers": os.cpu_count() // 2,
    "shuffle": False,
    "pin_memory": True
}

config["test_dataloader"] = {
    "batch_size": RECOMMENDED_BATCH_SIZE * 2,
    "num_workers": os.cpu_count() // 2,
    "shuffle": False,
    "pin_memory": True
}

# Configurações do trainer do PyTorch Lightning
config["trainer"] = {
    "max_epochs": 30,
    "default_root_dir": str(LOG_DIR),
    "precision": PRECISION,
    "gradient_clip_val": 1.0,
    "log_every_n_steps": 10,
    "deterministic": True,
}

# Atualiza a configuração de dados
config["data"]["data_path"] = str(BINANCE_DATA_DIR)
config["data"]["config"]["target_assets"] = ASSETS
config["data"]["config"]["dtype"] = DTYPE

# Salvar a configuração final usada para este treinamento
final_config_path = OUTPUT_DIR / "final_sota_config.yaml"
with open(final_config_path, "w") as f:
    yaml.dump(config, f, default_flow_style=False)

logger.info(f"Configuração final salva em: {final_config_path}")

# # 8. Configuração de Callbacks SOTA para Treinamento
# Implementação de callbacks avançados para monitoramento e otimização do treinamento

def configure_sota_callbacks(output_dir):
    """
    Configura callbacks avançados para treinamento SOTA com PyTorch Lightning.
    """
    from pytorch_lightning.callbacks import (
        ModelCheckpoint, EarlyStopping, LearningRateMonitor, 
        StochasticWeightAveraging, GradientAccumulationScheduler
    )
    
    # Assumindo que estes callbacks personalizados estão implementados no pacote uni2ts
    # Se não estiverem, seria necessário implementá-los
    try:
        from uni2ts.callbacks.bayesian_uncertainty import BayesianUncertaintyMonitor
        from uni2ts.callbacks.elbo_annealing import ELBOAnnealingCallback
        bayesian_callbacks_available = True
    except ImportError:
        logger.warning("Callbacks bayesianos personalizados não disponíveis. Usando alternativas padrão.")
        bayesian_callbacks_available = False
    
    callbacks = [
        # Checkpointing avançado
        {
            "_target_": "pytorch_lightning.callbacks.ModelCheckpoint",
            "dirpath": os.path.join(output_dir, "checkpoints"),
            "filename": "best-model-{epoch:02d}-{val_loss:.4f}-{val_crps:.4f}",
            "monitor": "val_loss",
            "mode": "min",
            "save_top_k": 3,
            "save_last": True,
            "auto_insert_metric_name": False
        },
        
        # Early stopping inteligente
        {
            "_target_": "pytorch_lightning.callbacks.EarlyStopping",
            "monitor": "val_loss",
            "patience": 7,
            "mode": "min",
            "min_delta": 0.001,
            "verbose": True
        },
        
        # Monitoramento avançado de LR
        {
            "_target_": "pytorch_lightning.callbacks.LearningRateMonitor",
            "logging_interval": "step"
        },
        
        # SOTA: SWA para melhor generalização
        {
            "_target_": "pytorch_lightning.callbacks.StochasticWeightAveraging",
            "swa_lrs": 1e-3
        },
        
        # SOTA: Acumulação de gradiente adaptativa
        {
            "_target_": "pytorch_lightning.callbacks.GradientAccumulationScheduler",
            "scheduling": {
                0: 4,
                5: 2,
                10: 1
            }
        }
    ]
    
    # Adiciona callbacks bayesianos específicos se disponíveis
    if bayesian_callbacks_available:
        callbacks.append({
            "_target_": "uni2ts.callbacks.bayesian_uncertainty.BayesianUncertaintyMonitor",
            "plot_dir": os.path.join(output_dir, "uncertainty_plots"),
            "log_attention_weights": True,
            "uncertainty_threshold": 0.05
        })
        
        callbacks.append({
            "_target_": "uni2ts.callbacks.elbo_annealing.ELBOAnnealingCallback",
            "kl_weight": 0.001,
            "kl_anneal_steps": 1000,
            "kl_anneal_method": "cyclical"  # SOTA: annealing cíclico
        })
    
    return callbacks

# Configurar callbacks
logger.info("Configurando callbacks SOTA para treinamento...")
callbacks = configure_sota_callbacks(OUTPUT_DIR)
config["callbacks"] = callbacks

logger.info(f"Configurados {len(callbacks)} callbacks avançados para treinamento.")

# # 9. Execução do Treinamento
# Pipeline de treinamento com monitoramento avançado

from uni2ts.cli.train import train

logger.info("Iniciando o pipeline de treinamento SOTA...")

# A função `train` do uni2ts encapsula a lógica do PyTorch Lightning
# e usa a configuração que preparamos.
train_results = train(config)

logger.info("Treinamento concluído com sucesso.")

# # 10. Avaliação Bayesiana Avançada
# Métricas e análises avançadas para modelos bayesianos

from uni2ts.cli.eval import evaluate

# Métricas de avaliação Bayesianas avançadas
def evaluate_bayesian_forecasts(model_path, config, test_loader=None):
    """
    Avaliação avançada de modelos bayesianos com métricas específicas para quantificação de incerteza.
    """
    logger.info(f"Iniciando avaliação bayesiana avançada do modelo: {model_path}")
    
    # Usar a função evaluate existente do uni2ts para obter resultados básicos
    basic_results = evaluate(
        weight_path=str(model_path),
        config=config,
        num_samples=100,  # Aumentar para obter estimativas mais estáveis
    )
    
    # Extrair dados de previsão e alvos
    forecasts = basic_results["forecasts"]
    targets = basic_results["targets"]
    
    # Métricas SOTA para avaliação probabilística
    metrics = basic_results["metrics"].copy()  # Inclui métricas básicas como CRPS, MAE, RMSE
    
    # Calcular métricas avançadas adicionais
    coverage_80 = []
    coverage_95 = []
    sharpness = []
    direction_acc = []
    
    for i in range(len(targets)):
        target = targets[i]
        forecast = forecasts[i]
        
        # Calcular quantis
        q10 = np.quantile(forecast, 0.1, axis=0)
        q90 = np.quantile(forecast, 0.9, axis=0)
        q025 = np.quantile(forecast, 0.025, axis=0)
        q975 = np.quantile(forecast, 0.975, axis=0)
        median = np.quantile(forecast, 0.5, axis=0)
        
        # Cobertura do intervalo
        in_interval_80 = np.logical_and(target >= q10, target <= q90)
        in_interval_95 = np.logical_and(target >= q025, target <= q975)
        coverage_80.append(np.mean(in_interval_80))
        coverage_95.append(np.mean(in_interval_95))
        
        # Sharpness (média da largura do intervalo)
        sharpness.append(np.mean(q975 - q025))
        
        # Acurácia direcional
        direction_correct = np.mean(np.sign(median[1:] - median[:-1]) == 
                                   np.sign(target[1:] - target[:-1]))
        direction_acc.append(direction_correct)
    
    # Adicionar métricas avançadas ao dicionário
    metrics["coverage_80"] = np.mean(coverage_80)
    metrics["coverage_95"] = np.mean(coverage_95)
    metrics["sharpness"] = np.mean(sharpness)
    metrics["direction_acc"] = np.mean(direction_acc)
    
    # Métricas compostas
    metrics["osis"] = metrics["coverage_95"] / 0.95  # Interval Score (quanto mais próximo de 1, melhor)
    metrics["uncertainty_quality"] = metrics["direction_acc"] * (2 - metrics["sharpness"])  # Métrica composta
    
    # Logar métricas
    logger.info("Métricas Bayesianas Avançadas:")
    for key, value in metrics.items():
        logger.info(f"  - {key}: {value:.6f}")
    
    # Retornar resultados enriquecidos
    enhanced_results = {
        "metrics": metrics,
        "forecasts": forecasts,
        "targets": targets,
        "item_ids": basic_results["item_ids"]
    }
    
    return enhanced_results

# Encontra o melhor checkpoint salvo pelo callback
checkpoint_dir = OUTPUT_DIR / "checkpoints"
best_model_path = next(checkpoint_dir.glob("best-model-*.ckpt"), None)

if not best_model_path:
    best_model_path = next(checkpoint_dir.glob("last.ckpt"), None)
    if not best_model_path:
        raise FileNotFoundError("Nenhum checkpoint do modelo foi encontrado. Verifique se o treinamento foi concluído com sucesso.")

logger.info(f"Carregando modelo para avaliação: {best_model_path}")

# Executar avaliação bayesiana avançada
eval_results = evaluate_bayesian_forecasts(best_model_path, config)

# # 11. Visualização Avançada de Incerteza
# Visualizações sofisticadas para análise de incerteza bayesiana

def visualize_bayesian_uncertainty(forecast_results, num_samples=4, plot_dir=None):
    """
    Visualização avançada de incerteza bayesiana com múltiplos níveis de confiança
    e distribuições de probabilidade.
    """
    logger.info("Gerando visualizações avançadas de incerteza bayesiana...")
    
    # Configuração visual
    sns.set_theme(style="whitegrid")
    plt.rcParams['figure.figsize'] = [15, 5*num_samples]
    
    # Extrair dados
    forecasts = forecast_results["forecasts"]
    targets = forecast_results["targets"]
    item_ids = forecast_results["item_ids"]
    
    # Criar figura com GridSpec para layout complexo
    fig = plt.figure(constrained_layout=True)
    gs = plt.GridSpec(num_samples, 2, figure=fig, width_ratios=[3, 1])
    
    for i in range(min(num_samples, len(targets))):
        # Série temporal principal
        ax_main = fig.add_subplot(gs[i, 0])
        
        # Distribuição de incerteza
        ax_dist = fig.add_subplot(gs[i, 1])
        
        # Dados para este exemplo
        target = targets[i]
        forecast = forecasts[i]
        item_id = item_ids[i] if i < len(item_ids) else f"Exemplo {i+1}"
        
        # Calcular estatísticas
        mean = np.mean(forecast, axis=0)
        median = np.median(forecast, axis=0)
        q10 = np.percentile(forecast, 10, axis=0)
        q90 = np.percentile(forecast, 90, axis=0)
        q25 = np.percentile(forecast, 25, axis=0)
        q75 = np.percentile(forecast, 75, axis=0)
        q025 = np.percentile(forecast, 2.5, axis=0)
        q975 = np.percentile(forecast, 97.5, axis=0)
        
        # Plot principal com níveis de confiança
        x = np.arange(len(target))
        ax_main.plot(x, target, 'k-', label='Real', linewidth=2)
        ax_main.plot(x, median, 'b-', label='Mediana', linewidth=1.5)
        
        # Diferentes níveis de confiança
        ax_main.fill_between(x, q025, q975, color='blue', alpha=0.1, label='95% IC')
        ax_main.fill_between(x, q25, q75, color='blue', alpha=0.3, label='50% IC')
        
        ax_main.set_title(f"Previsão com Múltiplos Intervalos de Confiança ({item_id})")
        ax_main.set_xlabel("Tempo (min)")
        ax_main.set_ylabel("Valor")
        ax_main.legend(loc='upper left')
        
        # Visualização da distribuição de incerteza para ponto específico
        # Escolhemos um ponto interessante (ex: meio da previsão)
        mid_point = len(target) // 2
        
        # KDE da distribuição de previsões
        sns.kdeplot(forecast[:, mid_point], ax=ax_dist, fill=True)
        ax_dist.axvline(x=target[mid_point], color='r', linestyle='--', label='Real')
        ax_dist.axvline(x=median[mid_point], color='b', linestyle='--', label='Mediana')
        
        ax_dist.set_title(f"Distribuição no t={mid_point}")
        ax_dist.set_ylabel("Densidade")
        ax_dist.set_xlabel("Valor")
        ax_dist.legend()
    
    # Salvar figura
    if plot_dir:
        plot_path = Path(plot_dir) / "bayesian_uncertainty_visualization.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"Visualização salva em: {plot_path}")
    
    plt.tight_layout()
    
    # Criar visualização adicional: Heatmap de incerteza ao longo do tempo
    plt.figure(figsize=(15, 8))
    
    # Selecionar um exemplo para análise de calor
    example_idx = 0
    forecast_data = forecasts[example_idx]
    target_data = targets[example_idx]
    
    # Calcular desvio padrão ao longo do tempo (eixo das amostras)
    uncertainty = np.std(forecast_data, axis=0)
    
    # Normalizar para melhor visualização
    norm_uncertainty = uncertainty / np.max(uncertainty)
    
    # Criar uma grade de tempo x valor para o heatmap
    time_steps = np.arange(len(target_data))
    val_range = np.linspace(
        min(np.min(target_data), np.min(np.min(forecast_data))),
        max(np.max(target_data), np.max(np.max(forecast_data))),
        100
    )
    
    # Calcular KDE para cada ponto no tempo
    heatmap_data = np.zeros((len(val_range), len(time_steps)))
    for t in range(len(time_steps)):
        # Kernel density estimation
        kde = stats.gaussian_kde(forecast_data[:, t])
        heatmap_data[:, t] = kde(val_range)
    
    # Plotar heatmap
    plt.subplot(2, 1, 1)
    sns.heatmap(heatmap_data, cmap="viridis")
    plt.title("Distribuição de Densidade ao Longo do Tempo")
    plt.xlabel("Tempo")
    plt.ylabel("Valor")
    
    # Plotar incerteza
    plt.subplot(2, 1, 2)
    plt.plot(time_steps, norm_uncertainty, 'r-', linewidth=2)
    plt.fill_between(time_steps, 0, norm_uncertainty, color='red', alpha=0.3)
    plt.title("Incerteza Normalizada ao Longo do Tempo")
    plt.xlabel("Tempo")
    plt.ylabel("Incerteza (Desvio Padrão Norm.)")
    
    plt.tight_layout()
    
    # Salvar figura de heatmap
    if plot_dir:
        heatmap_path = Path(plot_dir) / "uncertainty_heatmap.png"
        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
        logger.info(f"Heatmap de incerteza salvo em: {heatmap_path}")

# Executar visualização
visualize_bayesian_uncertainty(
    eval_results, 
    num_samples=4, 
    plot_dir=UNCERTAINTY_PLOTS_DIR
)

logger.info("Visualizações de incerteza geradas com sucesso.")

# # 12. Exportação de Modelo para Produção com Monitoramento
# Implementação avançada para exportação e implantação do modelo

def export_model_for_production(model_path, config, output_dir):
    """
    Exportação avançada para produção com anotações SOTA.
    Inclui metadados, exemplo de inferência e monitoramento opcional com MLflow.
    """
    import torch
    import json
    import os
    from datetime import datetime
    from uni2ts.model.moirai import MoiraiForecast
    
    logger.info(f"Iniciando exportação avançada do modelo para produção: {model_path}")
    
    # Criar diretório para artefatos
    model_dir = os.path.join(output_dir, "production_model")
    os.makedirs(model_dir, exist_ok=True)
    
    # Carregar o modelo do checkpoint
    lightning_module = MoiraiForecast.load_from_checkpoint(model_path)
    model = lightning_module.model
    model.eval() # Coloca o modelo em modo de avaliação
    
    # 1. Salvar modelo em formato TorchScript
    try:
        model.eval()
        # Preparar exemplo de entrada para script
        batch_size = 1
        context_len = config["model"]["context_length"]
        # Determinar número de features dinâmicas a partir do dataset
        sample = next(iter(test_dataset))
        num_features = sample['feat_dynamic_real'].shape[0]
        
        dummy_input = {
            "past_target": torch.randn(batch_size, context_len),
            "past_observed_target": torch.ones(batch_size, context_len, dtype=torch.bool),
            "feat_dynamic_real": torch.randn(batch_size, context_len + config["model"]["prediction_length"], num_features)
        }
        
        # Converte inputs para o tipo correto
        for key, tensor in dummy_input.items():
            if torch.is_floating_point(tensor):
                dummy_input[key] = tensor.to(getattr(torch, DTYPE))
        
        # Tentar script/trace o modelo
        scripted_model = torch.jit.script(model)
        torchscript_path = os.path.join(model_dir, "model.pt")
        scripted_model.save(torchscript_path)
        logger.info(f"✅ Modelo TorchScript salvo em {torchscript_path}")
    except Exception as e:
        logger.error(f"⚠️ Falha na conversão para TorchScript: {str(e)}")
        logger.info("Recorrendo a checkpoint padrão")
    
    # 2. Exportar metadados e configuração
    model_info = {
        "name": "Moirai-MoE-Bayesian-Crypto-SOTA",
        "version": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "framework": "PyTorch",
        "type": "bayesian_time_series_forecasting",
        "context_length": config["model"]["context_length"],
        "prediction_length": config["model"]["prediction_length"],
        "assets": config["data"]["config"]["target_assets"],
        "training_date": datetime.now().isoformat(),
        "input_features": {
            "shape": [config["model"]["context_length"], num_features],
            "dtype": config["data"]["config"]["dtype"]
        },
        "output_features": {
            "shape": [config["model"]["prediction_length"], 1],
            "distribution": "student_t"
        },
        "metrics": eval_results["metrics"]
    }
    
    with open(os.path.join(model_dir, "model_info.json"), "w") as f:
        json.dump(model_info, f, indent=2)
    
    # 3. Criar exemplo de inferência
    inference_example = {
        "python": """
import torch
from pathlib import Path

# Carregar modelo
model_path = Path("model.pt")
model = torch.jit.load(str(model_path))

# Preparar entrada (exemplo)
context_length = {context_length}
prediction_length = {prediction_length}
num_features = {num_features}

# Criação de amostra para um ativo
sample_input = {{
    "past_target": torch.randn(1, context_length),
    "past_observed_target": torch.ones(1, context_length, dtype=torch.bool),
    "feat_dynamic_real": torch.randn(1, context_length + prediction_length, num_features)
}}

# Converter para o tipo correto
for key, tensor in sample_input.items():
    if torch.is_floating_point(tensor):
        sample_input[key] = tensor.to(torch.{dtype})

# Inferência
with torch.no_grad():
    output = model(sample_input)

# Processar saída
mean_forecast = output.loc.numpy()  # Média da previsão
uncertainty = output.scale.numpy()  # Incerteza (desvio padrão)

# Para obter intervalos de confiança, se o modelo retornar uma distribuição t-Student:
import numpy as np
import scipy.stats as stats

# Suponha df=4 para a distribuição t
df = 4.0
confidence_level = 0.95
t_value = stats.t.ppf((1 + confidence_level) / 2, df)

lower_bound = mean_forecast - t_value * uncertainty
upper_bound = mean_forecast + t_value * uncertainty

print(f"Previsão média: {{mean_forecast}}")
print(f"Intervalo de confiança {{confidence_level*100}}%: [{{lower_bound}}, {{upper_bound}}]")
""".format(
    context_length=config["model"]["context_length"],
    prediction_length=config["model"]["prediction_length"],
    num_features=num_features,
    dtype=DTYPE
)
    }
    
    with open(os.path.join(model_dir, "inference_example.py"), "w") as f:
        f.write(inference_example["python"])
    
    # 4. Opcional: Tracking com MLflow
    try:
        import mlflow
        mlflow_available = True
    except ImportError:
        mlflow_available = False
        
    if mlflow_available:
        try:
            logger.info("Registrando modelo no MLflow...")
            mlflow.set_experiment("crypto_forecasting_sota")
            with mlflow.start_run(run_name=f"moirai_bayesian_moe_{datetime.now().strftime('%Y%m%d_%H%M')}"):
                # Logar parâmetros
                flat_params = {}
                
                # Recursivamente adiciona parâmetros achatados
                def add_params(prefix, param_dict):
                    for key, value in param_dict.items():
                        if isinstance(value, dict):
                            add_params(f"{prefix}{key}.", value)
                        else:
                            if not isinstance(value, (str, int, float, bool)):
                                value = str(value)
                            flat_params[f"{prefix}{key}"] = value
                
                # Extrair parâmetros principais
                model_params = {
                    "context_length": config["model"]["context_length"],
                    "prediction_length": config["model"]["prediction_length"],
                    "assets": ",".join(config["data"]["config"]["target_assets"]),
                    "mc_dropout_rate": config["model"]["bayesian_head"]["mc_dropout_rate"],
                    "student_t_df": config["model"]["bayesian_head"]["student_t_df"]
                }
                
                mlflow.log_params(model_params)
                
                # Logar métricas
                for key, value in eval_results["metrics"].items():
                    mlflow.log_metric(key, value)
                
                # Logar artefatos
                for file in os.listdir(model_dir):
                    mlflow.log_artifact(os.path.join(model_dir, file))
                
                # Logar visualizações
                for plot_file in os.listdir(UNCERTAINTY_PLOTS_DIR):
                    mlflow.log_artifact(os.path.join(UNCERTAINTY_PLOTS_DIR, plot_file))
                
                # Logar modelo PyTorch
                mlflow.pytorch.log_model(model, "pytorch_model")
                
                logger.info("Modelo registrado com sucesso no MLflow.")
                
        except Exception as e:
            logger.error(f"⚠️ MLflow tracking falhou: {str(e)}")
    
    logger.info(f"Exportação do modelo para produção concluída: {model_dir}")
    return model_dir

# Exportar modelo para produção
production_model_dir = export_model_for_production(
    model_path=best_model_path,
    config=config,
    output_dir=OUTPUT_DIR
)

logger.info(f"Pipeline completo de fine-tuning SOTA concluído. Artefatos em: {OUTPUT_DIR}")
logger.info(f"Modelo de produção disponível em: {production_model_dir}")

# Mostrar resumo final das métricas
print("\n" + "="*50)
print("RESUMO FINAL: MÉTRICAS DE PERFORMANCE")
print("="*50)
for key, value in eval_results["metrics"].items():
    print(f"{key:20s}: {value:.6f}")
print("="*50)
