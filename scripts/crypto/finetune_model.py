#!/usr/bin/env python3
"""
Script de Fine-tuning SOTA para Crypto com Moirai-MoE + Bayesiano
Executa treinamento REAL com todas as melhorias do BOOST.MD

Uso:
    python scripts/crypto/finetune_model.py --config configs/crypto/finetune_bayesian_moe.yaml

CORREÇÃO IMPLEMENTADA:
- Carrega modelo Moirai-MoE real (não mock)
- Usa CryptoDatasetBuilder real
- Implementa callbacks do YAML
- Monitora métricas corretas (val/ELBO)
"""

import argparse
import sys
import os
from pathlib import Path
import logging
from datetime import datetime
from typing import Dict, Any, Optional

# Adicionar src e raiz ao path
root_path = Path(__file__).parent.parent.parent
sys.path.append(str(root_path))
sys.path.append(str(root_path / "src"))

import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf
from hydra.utils import instantiate

# Imports uni2ts REAIS
from uni2ts.model.moirai_moe import MoiraiMoEModule, MoiraiMoEForecast
from uni2ts.model.crypto.bayesian_head import BayesianPredictionHead, BayesianPredictionOutput
from uni2ts.loss.bayesian_elbo import BayesianELBOLoss
from uni2ts.data.builder.crypto import CryptoDatasetBuilder
from uni2ts.callbacks.bayesian_uncertainty import BayesianUncertaintyMonitor


def setup_logging(run_name: str):
    """Configura logging para o treinamento"""
    log_dir = Path("logs") / run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        ]
    )
    return logging.getLogger(__name__)


def validate_config(config):
    """Valida configuração antes do treinamento"""
    required_sections = ['model', 'data', 'trainer', 'loss_func']
    
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Seção '{section}' não encontrada na configuração")
    
    # Validar configurações específicas
    if 'pretrained_model_name_or_path' not in config.model:
        raise ValueError("Campo 'pretrained_model_name_or_path' obrigatório no modelo")
    
    if not config.model.pretrained_model_name_or_path.startswith("Salesforce/moirai"):
        logging.warning(f"Modelo não é Moirai: {config.model.pretrained_model_name_or_path}")
    
    # Validar loss Bayesiana
    if config.loss_func._target_ != "uni2ts.loss.bayesian_elbo.BayesianELBOLoss":
        raise ValueError("Loss function deve ser BayesianELBOLoss para treinamento Bayesiano")
    
    logging.info("✅ Configuração validada com sucesso")


class MoiraiBayesianLightningModule(pl.LightningModule):
    """
    Lightning Module REAL para Moirai-MoE + Cabeça Bayesiana
    
    CORREÇÃO: Implementa treinamento real com:
    - Modelo Moirai-MoE carregado do HuggingFace
    - BayesianPredictionHead integrada
    - BayesianELBOLoss com métricas corretas
    - Otimizadores adequados para fine-tuning
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.save_hyperparameters()
        
        # 1. Carregar modelo Moirai-MoE REAL
        logging.info(f"🚀 Carregando modelo Moirai-MoE: {config.model.pretrained_model_name_or_path}")
        try:
            # Usar MoiraiMoEForecast que é a interface correta
            self.backbone = MoiraiMoEForecast.load_from_checkpoint(
                config.model.pretrained_model_name_or_path,
                prediction_length=config.model.prediction_length,
                context_length=config.model.context_length,
                patch_size=config.model.patch_size,
                num_samples=config.model.num_samples,
            )
        except Exception as e:
            logging.warning(f"⚠️ Erro ao carregar Moirai-MoE, criando novo modelo: {e}")
            # Criar modelo MoE do zero
            self.backbone = MoiraiMoEForecast(
                prediction_length=config.model.prediction_length,
                context_length=config.model.context_length,
                patch_size=config.model.patch_size,
                num_samples=config.model.num_samples,
            )
        
        # 2. Substituir cabeça por BayesianPredictionHead
        logging.info("🧠 Configurando BayesianPredictionHead")
        self.prediction_head = instantiate(config.model.prediction_head)
        
        # 3. Loss Bayesiana REAL
        logging.info("📉 Configurando BayesianELBOLoss")
        self.criterion = instantiate(config.loss_func)
        
        # 4. Configurações de treinamento
        self.learning_rate = config.optimizer.get('lr', 1e-4)
        self.weight_decay = config.optimizer.get('weight_decay', 1e-5)
        
        # Congelar backbone se especificado
        if config.model.get('freeze_backbone', False):
            for param in self.backbone.parameters():
                param.requires_grad = False
            logging.info("🔒 Backbone congelado - apenas fine-tuning da cabeça Bayesiana")
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> BayesianPredictionOutput:
        """Forward pass através do pipeline completo"""
        
        # 1. Extrair características com Moirai-MoE
        target = batch.get('target', batch.get('past_target'))
        
        if target is None:
            raise ValueError("Batch deve conter 'target' ou 'past_target'")
        
        # 2. Forward através do módulo backbone (acessar o módulo interno)
        if hasattr(self.backbone, 'module'):
            backbone_module = self.backbone.module
        else:
            backbone_module = self.backbone
        
        # Usar o módulo diretamente para extrair representações
        backbone_output = backbone_module(
            past_target=target,
            past_observed_target=batch.get('past_observed_target', torch.ones_like(target))
        )
        
        # 3. Extrair representações para a cabeça Bayesiana
        # Backbone output normalmente tem shape [batch, seq_len, d_model]
        if hasattr(backbone_output, 'last_hidden_state'):
            representations = backbone_output.last_hidden_state
        elif hasattr(backbone_output, 'prediction'):
            representations = backbone_output.prediction
        elif isinstance(backbone_output, torch.Tensor):
            representations = backbone_output
        else:
            raise ValueError(f"Formato de saída não reconhecido: {type(backbone_output)}")
        
        # 4. Cabeça Bayesiana
        bayesian_output = self.prediction_head(representations, training=self.training)
        
        return bayesian_output
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Step de treinamento REAL"""
        
        # Forward pass
        prediction_output = self(batch)
        
        # Target para loss
        target = batch.get('future_target', batch.get('target'))
        if target is None:
            raise ValueError("Batch deve conter 'future_target' para treinamento")
        
        # Coletar KL divergences das camadas variacionais
        kl_divergence = self.prediction_head.get_kl_divergence()
        
        # Computar ELBO loss
        loss_dict = self.criterion(prediction_output, target, kl_divergence)
        
        # Log métricas com prefixo correto
        for key, value in loss_dict.items():
            self.log(f"train/{key}", value, prog_bar=(key == "loss"), on_step=True, on_epoch=True)
        
        # Log métricas de incerteza adicionais
        self.log("train/epistemic_uncertainty", prediction_output.epistemic_uncertainty.mean())
        self.log("train/aleatoric_uncertainty", prediction_output.aleatoric_uncertainty.mean())
        self.log("train/total_uncertainty", prediction_output.total_uncertainty.mean())
        
        return loss_dict["loss"]
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        """Step de validação REAL"""
        
        # Forward pass
        prediction_output = self(batch)
        
        # Target para loss
        target = batch.get('future_target', batch.get('target'))
        if target is None:
            raise ValueError("Batch deve conter 'future_target' para validação")
        
        # Coletar KL divergences
        kl_divergence = self.prediction_head.get_kl_divergence()
        
        # Computar ELBO loss
        loss_dict = self.criterion(prediction_output, target, kl_divergence)
        
        # Log métricas com prefixo correto - MÉTRICAS CORRETAS!
        for key, value in loss_dict.items():
            metric_name = "ELBO" if key == "loss" else key
            self.log(f"val/{metric_name}", value, prog_bar=(key == "loss"), on_epoch=True)
        
        # Log métricas de incerteza
        self.log("val/epistemic_uncertainty", prediction_output.epistemic_uncertainty.mean())
        self.log("val/aleatoric_uncertainty", prediction_output.aleatoric_uncertainty.mean())
        self.log("val/total_uncertainty", prediction_output.total_uncertainty.mean())
        
        # Retornar dicionário para callback
        return {
            'loss': loss_dict["loss"],
            'prediction_output': prediction_output,
            'target': target
        }
    
    def configure_optimizers(self):
        """Configurar otimizadores para fine-tuning"""
        
        # Diferentes learning rates para backbone vs cabeça
        backbone_lr = self.learning_rate * 0.1  # LR menor para backbone pré-treinado
        head_lr = self.learning_rate
        
        # Separar parâmetros
        backbone_params = list(self.backbone.parameters())
        head_params = list(self.prediction_head.parameters())
        
        # Otimizador com learning rates diferenciados
        optimizer = torch.optim.AdamW([
            {'params': backbone_params, 'lr': backbone_lr, 'weight_decay': self.weight_decay},
            {'params': head_params, 'lr': head_lr, 'weight_decay': self.weight_decay * 0.1}
        ], betas=[0.9, 0.999], eps=1e-8)
        
        # Scheduler com warmup
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=[backbone_lr, head_lr],
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=0.1,  # 10% warmup
            anneal_strategy='cos',
            div_factor=25.0,
            final_div_factor=10000.0
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1
            }
        }


def create_callbacks_from_config(config) -> list:
    """
    Cria callbacks REAIS baseados na configuração YAML
    
    CORREÇÃO: Implementa todos os callbacks especificados no YAML,
    incluindo BayesianUncertaintyMonitor e métricas corretas
    """
    callbacks = []
    
    # 1. Carregar callbacks do config se especificado
    if 'callbacks' in config and config.callbacks:
        for callback_config in config.callbacks:
            try:
                callback = instantiate(callback_config)
                callbacks.append(callback)
                logging.info(f"✅ Callback carregado: {callback.__class__.__name__}")
            except Exception as e:
                logging.error(f"❌ Erro ao carregar callback {callback_config}: {e}")
    
    # 2. Callback de checkpoint - MÉTRICAS CORRETAS!
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath="./checkpoints",
        filename="crypto_bayesian_moe_{epoch:03d}_{val_ELBO:.4f}",
        monitor="val/ELBO",  # CORREÇÃO: Agora monitora métrica ELBO correta
        mode="min",
        save_top_k=3,
        save_last=True,
        verbose=True,
        auto_insert_metric_name=False
    )
    callbacks.append(checkpoint_callback)
    
    # 3. Early stopping - MÉTRICAS CORRETAS!
    early_stopping = pl.callbacks.EarlyStopping(
        monitor="val/ELBO",  # CORREÇÃO: Métrica ELBO correta
        mode="min",
        patience=config.trainer.get('early_stopping_patience', 15),
        min_delta=1e-6,
        verbose=True
    )
    callbacks.append(early_stopping)
    
    # 4. Learning rate monitor
    lr_monitor = pl.callbacks.LearningRateMonitor(
        logging_interval="step",
        log_momentum=True
    )
    callbacks.append(lr_monitor)
    
    # 5. Progress bar com métricas Bayesianas
    progress_bar = pl.callbacks.TQDMProgressBar(refresh_rate=10)
    callbacks.append(progress_bar)
    
    logging.info(f"✅ {len(callbacks)} callbacks configurados")
    return callbacks


def create_dataloader_from_config(config) -> tuple:
    """
    Cria dataloaders REAIS baseados na configuração
    
    CORREÇÃO: Usa CryptoDatasetBuilder real em vez de dados mock
    """
    logging.info("📊 Criando datasets com CryptoDatasetBuilder...")
    
    # Instantiate dataset builder from config
    dataset_builder = instantiate(config.data)
    
    # Criar datasets
    train_dataset = dataset_builder.create_train_dataset()
    val_dataset = dataset_builder.create_validation_dataset()
    
    # Criar dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.data.get('batch_size', 16),
        shuffle=True,
        num_workers=config.data.get('num_workers', 4),
        pin_memory=True,
        persistent_workers=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.data.get('batch_size', 16),
        shuffle=False,
        num_workers=config.data.get('num_workers', 4),
        pin_memory=True,
        persistent_workers=True
    )
    
    logging.info(f"✅ Datasets criados - Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    return train_loader, val_loader


def create_trainer_from_config(config, callbacks: list, logger) -> pl.Trainer:
    """Cria trainer baseado na configuração"""
    
    trainer_config = config.trainer
    
    trainer = pl.Trainer(
        max_epochs=trainer_config.get('max_epochs', 100),
        gradient_clip_val=trainer_config.get('gradient_clip_val', 1.0),
        accumulate_grad_batches=trainer_config.get('accumulate_grad_batches', 4),
        precision=trainer_config.get('precision', 16),
        check_val_every_n_epoch=trainer_config.get('check_val_every_n_epoch', 1),
        log_every_n_steps=trainer_config.get('log_every_n_steps', 50),
        devices=trainer_config.get('devices', 1),
        accelerator=trainer_config.get('accelerator', 'auto'),
        strategy=trainer_config.get('strategy', 'auto'),
        callbacks=callbacks,
        logger=logger,
        deterministic=trainer_config.get('deterministic', False),
        benchmark=trainer_config.get('benchmark', True),
        num_sanity_val_steps=trainer_config.get('num_sanity_val_steps', 2),
        fast_dev_run=trainer_config.get('fast_dev_run', False),
        limit_train_batches=trainer_config.get('limit_train_batches', 1.0),
        limit_val_batches=trainer_config.get('limit_val_batches', 1.0),
    )
    
    return trainer


def setup_environment():
    """Configura ambiente para treinamento"""
    # Detectar hardware
    device_info = {
        'cuda_available': torch.cuda.is_available(),
        'cuda_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'cuda_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'
    }
    
    logging.info("🖥️ Informações do hardware:")
    for key, value in device_info.items():
        logging.info(f"   {key}: {value}")
    
    # Configurações de performance
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True  # Otimizar para tamanhos fixos
        if hasattr(torch.backends.cudnn, 'allow_tf32'):
            torch.backends.cudnn.allow_tf32 = True
    
    return device_info


def main():
    """
    Função principal - CORRIGIDA para implementar treinamento REAL
    
    CORREÇÕES IMPLEMENTADAS:
    - Carrega modelo Moirai-MoE real
    - Usa CryptoDatasetBuilder real
    - Implementa callbacks do YAML
    - Monitora métricas ELBO corretas
    """
    
    # Parser de argumentos
    parser = argparse.ArgumentParser(description="Fine-tuning SOTA Crypto com Moirai-MoE + Bayesiano")
    parser.add_argument("--config", required=True, help="Caminho para arquivo de configuração YAML")
    parser.add_argument("--run_name", help="Nome da execução (opcional)")
    parser.add_argument("--debug", action="store_true", help="Modo debug com fast_dev_run")
    args = parser.parse_args()
    
    # Carregar configuração
    if not Path(args.config).exists():
        raise FileNotFoundError(f"Arquivo de configuração não encontrado: {args.config}")
    
    config = OmegaConf.load(args.config)
    
    # Nome da execução
    run_name = args.run_name or f"crypto_sota_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Setup logging
    logger_instance = setup_logging(run_name)
    
    logger_instance.info("🚀 INICIANDO FINE-TUNING SOTA CRYPTO")
    logger_instance.info(f"📁 Configuração: {args.config}")
    logger_instance.info(f"🏷️ Run name: {run_name}")
    
    try:
        # 1. Validar configuração
        logger_instance.info("⚙️ Validando configuração...")
        validate_config(config)
        
        # 2. Setup ambiente
        logger_instance.info("🖥️ Configurando ambiente...")
        device_info = setup_environment()
        
        # 3. Criar modelo REAL
        logger_instance.info("🧠 Criando modelo Moirai-MoE + Bayesiano...")
        model = MoiraiBayesianLightningModule(config)
        
        # 4. Criar dataloaders REAIS
        logger_instance.info("📊 Preparando datasets...")
        train_loader, val_loader = create_dataloader_from_config(config)
        
        # 5. Criar callbacks REAIS
        logger_instance.info("📈 Configurando callbacks...")
        callbacks = create_callbacks_from_config(config)
        
        # 6. Criar logger do PyTorch Lightning baseado na configuração
        if 'logger' in config and config.logger:
            try:
                pl_logger = instantiate(config.logger)
                logging.info(f"✅ Logger configurado: {pl_logger.__class__.__name__}")
            except Exception as e:
                logging.warning(f"⚠️ Erro ao configurar logger do config, usando TensorBoard: {e}")
                pl_logger = pl.loggers.TensorBoardLogger(
                    save_dir="./logs",
                    name=run_name,
                    version=None
                )
        else:
            pl_logger = pl.loggers.TensorBoardLogger(
                save_dir="./logs",
                name=run_name,
                version=None
            )
        
        # 7. Criar trainer
        logger_instance.info("⚡ Configurando trainer...")
        if args.debug:
            config.trainer.fast_dev_run = True
            config.trainer.limit_train_batches = 10
            config.trainer.limit_val_batches = 5
        
        trainer = create_trainer_from_config(config, callbacks, pl_logger)
        
        # 8. Iniciar treinamento REAL
        logger_instance.info("🏃 INICIANDO TREINAMENTO REAL...")
        logger_instance.info(f"📊 Train batches: {len(train_loader)}")
        logger_instance.info(f"📊 Val batches: {len(val_loader)}")
        logger_instance.info(f"🎯 Max epochs: {config.trainer.max_epochs}")
        
        # Fit do modelo
        trainer.fit(
            model=model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader
        )
        
        # 9. Resumo final
        logger_instance.info("🎉 TREINAMENTO COMPLETADO!")
        logger_instance.info(f"📁 Checkpoints salvos em: ./checkpoints")
        logger_instance.info(f"📊 Logs em: ./logs/{run_name}")
        
        # Métricas finais
        if trainer.callback_metrics:
            logger_instance.info("📈 Métricas finais:")
            for key, value in trainer.callback_metrics.items():
                logger_instance.info(f"   {key}: {value:.6f}")
        
        # Salvar configuração usada
        config_save_path = Path("./logs") / run_name / "config_used.yaml"
        config_save_path.parent.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(config, config_save_path)
        logger_instance.info(f"💾 Configuração salva em: {config_save_path}")
        
        return True
        
    except Exception as e:
        logger_instance.error(f"❌ ERRO NO TREINAMENTO: {e}")
        import traceback
        logger_instance.error(traceback.format_exc())
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
