"""
Integração CLI para Fine-tuning Crypto Bayesiano
Adapta o CLI uni2ts para usar componentes Bayesianos SOTA

SOTA implementado:
- Integração com BayesianPredictionHead
- Suporte para BayesianELBOLoss
- Callbacks de monitoramento de incerteza
- Métricas Bayesianas automáticas
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl

# Importar componentes SOTA
from uni2ts.model.crypto.bayesian_head import BayesianPredictionHead
from uni2ts.loss.bayesian_elbo import BayesianELBOLoss
from uni2ts.data.builder.crypto import CryptoDatasetBuilder
from uni2ts.callbacks.bayesian_uncertainty import BayesianUncertaintyMonitor
from uni2ts.eval_util.bayesian_metrics import create_bayesian_metrics


def setup_crypto_bayesian_model(cfg: DictConfig) -> pl.LightningModule:
    """
    Configura modelo Bayesiano SOTA para crypto
    
    Args:
        cfg: Configuração Hydra
        
    Returns:
        Modelo Lightning configurado
    """
    
    # Verificar se é configuração crypto Bayesiana
    if not cfg.get('crypto_specific', {}).get('bayesian_mode', True):
        raise ValueError("Esta função requer configuração Bayesiana (crypto_specific.bayesian_mode=true)")
    
    # Importar classe do modelo base (Moirai ou MoiraiFineture)
    if cfg.model.get('_target_'):
        model_cls = hydra.utils.get_class(cfg.model._target_)
    else:
        # Fallback para Moirai padrão
        from uni2ts.model.moirai import MoiraiForecast
        model_cls = MoiraiForecast
    
    # Configurar cabeça Bayesiana se especificada
    if cfg.model.get('prediction_head'):
        # Instanciar cabeça Bayesiana
        prediction_head = hydra.utils.instantiate(cfg.model.prediction_head)
        
        # Integrar com modelo base
        model_kwargs = OmegaConf.to_container(cfg.model, resolve=True)
        model_kwargs.pop('prediction_head', None)  # Remove para evitar conflito
        model_kwargs['prediction_head'] = prediction_head
    else:
        model_kwargs = OmegaConf.to_container(cfg.model, resolve=True)
    
    # Instanciar modelo
    model = hydra.utils.instantiate(cfg.model, **model_kwargs)
    
    return model


def setup_crypto_bayesian_callbacks(cfg: DictConfig) -> list:
    """
    Configura callbacks SOTA para monitoramento Bayesiano
    
    Args:
        cfg: Configuração Hydra
        
    Returns:
        Lista de callbacks configurados
    """
    
    callbacks = []
    
    # Callbacks padrão do config
    if cfg.get('callbacks'):
        for callback_cfg in cfg.callbacks:
            callbacks.append(hydra.utils.instantiate(callback_cfg))
    
    # Adicionar callback de monitoramento de incerteza se não existir
    has_uncertainty_monitor = any(
        isinstance(cb, BayesianUncertaintyMonitor) for cb in callbacks
    )
    
    if not has_uncertainty_monitor:
        # Usar configurações do config ou padrões
        uncertainty_config = cfg.get('crypto_specific', {}).get('uncertainty_monitoring', {})
        
        uncertainty_monitor = BayesianUncertaintyMonitor(
            log_attention_weights=uncertainty_config.get('attention_analysis', True),
            uncertainty_threshold=uncertainty_config.get('uncertainty_threshold', 0.05),
            log_interval=uncertainty_config.get('log_interval', 100),
            save_plots=True,
            plot_dir="./uncertainty_plots"
        )
        callbacks.append(uncertainty_monitor)
    
    return callbacks


def setup_crypto_bayesian_metrics(cfg: DictConfig) -> Dict[str, Any]:
    """
    Configura métricas Bayesianas SOTA
    
    Args:
        cfg: Configuração Hydra
        
    Returns:
        Dict com métricas configuradas
    """
    
    # Métricas do config
    config_metrics = []
    if cfg.get('metrics'):
        for metric_cfg in cfg.metrics:
            config_metrics.append(hydra.utils.instantiate(metric_cfg))
    
    # Adicionar métricas Bayesianas
    bayesian_metrics = create_bayesian_metrics()
    
    return {
        'config_metrics': config_metrics,
        'bayesian_metrics': bayesian_metrics
    }


def create_crypto_bayesian_trainer(cfg: DictConfig) -> pl.Trainer:
    """
    Cria trainer Lightning otimizado para crypto Bayesiano
    
    Args:
        cfg: Configuração Hydra
        
    Returns:
        Trainer configurado
    """
    
    # Callbacks Bayesianos
    callbacks = setup_crypto_bayesian_callbacks(cfg)
    
    # Logger
    logger = None
    if cfg.get('logger'):
        logger = hydra.utils.instantiate(cfg.logger)
    
    # Configurações do trainer
    trainer_kwargs = OmegaConf.to_container(cfg.trainer, resolve=True)
    trainer_kwargs['callbacks'] = callbacks
    
    if logger:
        trainer_kwargs['logger'] = logger
    
    # Criar trainer
    trainer = pl.Trainer(**trainer_kwargs)
    
    return trainer


def validate_crypto_bayesian_config(cfg: DictConfig) -> None:
    """
    Valida configuração para fine-tuning crypto Bayesiano
    
    Args:
        cfg: Configuração Hydra
        
    Raises:
        ValueError: Se configuração inválida
    """
    
    # Verificar componentes essenciais
    required_components = [
        'model',
        'data', 
        'trainer'
    ]
    
    for component in required_components:
        if component not in cfg:
            raise ValueError(f"Componente obrigatório ausente: {component}")
    
    # Verificar se é configuração Bayesiana
    if cfg.model.get('prediction_head', {}).get('_target_'):
        head_target = cfg.model.prediction_head._target_
        if 'BayesianPredictionHead' not in head_target:
            logging.warning(f"Cabeça de predição não é Bayesiana: {head_target}")
    
    # Verificar loss function
    if cfg.model.get('loss_func', {}).get('_target_'):
        loss_target = cfg.model.loss_func._target_
        if 'BayesianELBOLoss' not in loss_target:
            logging.warning(f"Loss function não é Bayesiana: {loss_target}")
    
    # Verificar dataset builder
    if cfg.data.get('_target_'):
        data_target = cfg.data._target_
        if 'CryptoDatasetBuilder' not in data_target:
            logging.warning(f"Dataset builder não é crypto-específico: {data_target}")
    
    logging.info("Configuração Bayesiana validada com sucesso")


def run_crypto_bayesian_training(cfg: DictConfig) -> Dict[str, Any]:
    """
    Executa fine-tuning crypto Bayesiano SOTA
    
    Args:
        cfg: Configuração Hydra
        
    Returns:
        Dict com resultados do treinamento
    """
    
    print("🚀 Iniciando Fine-tuning Crypto Bayesiano SOTA...")
    
    # Validar configuração
    validate_crypto_bayesian_config(cfg)
    
    # Setup componentes
    print("📊 Configurando modelo Bayesiano...")
    model = setup_crypto_bayesian_model(cfg)
    
    print("🔧 Configurando métricas...")
    metrics = setup_crypto_bayesian_metrics(cfg)
    
    print("⚡ Configurando trainer...")
    trainer = create_crypto_bayesian_trainer(cfg)
    
    # Setup dados
    print("📂 Carregando dataset...")
    datamodule = hydra.utils.instantiate(cfg.data)
    
    # Executar treinamento
    print("🎯 Iniciando treinamento...")
    trainer.fit(model, datamodule)
    
    # Resultados
    results = {
        'best_model_path': trainer.checkpoint_callback.best_model_path if trainer.checkpoint_callback else None,
        'final_metrics': {},  # Será preenchido pelos callbacks
        'uncertainty_summary': {}  # Será preenchido pelo UncertaintyMonitor
    }
    
    # Extrair métricas dos callbacks
    for callback in trainer.callbacks:
        if isinstance(callback, BayesianUncertaintyMonitor):
            results['uncertainty_summary'] = callback.get_uncertainty_summary()
    
    print("✅ Fine-tuning concluído com sucesso!")
    return results


# Função principal para uso com Hydra CLI
@hydra.main(version_base=None, config_path="../configs", config_name="crypto/finetune_bayesian_moe")
def main_crypto_bayesian(cfg: DictConfig) -> None:
    """
    Função principal para CLI de fine-tuning crypto Bayesiano
    
    Uso:
        python -m uni2ts.cli.crypto_bayesian
    """
    
    # Configurar ambiente
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
    
    # Executar treinamento
    results = run_crypto_bayesian_training(cfg)
    
    # Log resultados finais
    print("\n" + "="*50)
    print("📊 RESULTADOS FINAIS:")
    print("="*50)
    
    if results['best_model_path']:
        print(f"💾 Melhor modelo salvo em: {results['best_model_path']}")
    
    if results['uncertainty_summary']:
        print("🎯 Resumo de Incerteza:")
        for key, value in results['uncertainty_summary'].items():
            print(f"   {key}: {value}")
    
    print("\n🎉 Fine-tuning Bayesiano SOTA concluído!")


if __name__ == "__main__":
    main_crypto_bayesian()
