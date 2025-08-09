#!/usr/bin/env python3
"""
Script de Fine-tuning SOTA para Crypto com Moirai-MoE + Bayesiano
Executa treinamento com todas as melhorias do BOOST.MD

Uso:
    python scripts/crypto/finetune_model.py --config configs/crypto/finetune_bayesian_moe.yaml
"""

import argparse
import sys
import os
from pathlib import Path
import logging
from datetime import datetime

# Adicionar src e raiz ao path
root_path = Path(__file__).parent.parent.parent
sys.path.append(str(root_path))
sys.path.append(str(root_path / "src"))

import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf

# Imports uni2ts
from uni2ts.model.crypto.bayesian_head import BayesianPredictionHead
from uni2ts.loss.bayesian_elbo import BayesianELBOLoss
from uni2ts.data.builder.crypto import CryptoDatasetBuilder


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
            raise ValueError(f"Seção obrigatória '{section}' não encontrada no config")
    
    # Validar modelo Bayesiano
    if 'prediction_head' not in config.model:
        raise ValueError("prediction_head Bayesiano não configurado")
    
    # Validar loss Bayesiano
    if 'BayesianELBOLoss' not in config.loss_func._target_:
        raise ValueError("BayesianELBOLoss não configurada")
    
    # Validar dados
    if not config.data.config.unified_dataset:
        logging.warning("⚠️ Dataset não unificado - pode não ser SOTA")
    
    if not config.data.config.anonymous_training:
        logging.warning("⚠️ Treinamento não anônimo - pode não ser SOTA")


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


def create_trainer(config, logger=None):
    """Cria trainer Lightning com configurações SOTA"""
    
    # Callbacks
    callbacks = []
    if 'callbacks' in config:
        for callback_config in config.callbacks:
            # Implementar criação de callbacks baseado em config
            # Por simplicidade, usar callbacks básicos
            pass
    
    # Checkpoint callback básico
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath="./checkpoints",
        filename="crypto_bayesian_moe_{epoch:03d}_{val_loss:.4f}",
        monitor="val/loss",  # Mudar para val/ELBO quando disponível
        mode="min",
        save_top_k=3,
        save_last=True,
        verbose=True
    )
    callbacks.append(checkpoint_callback)
    
    # Early stopping
    early_stopping = pl.callbacks.EarlyStopping(
        monitor="val/loss",  # Mudar para val/ELBO quando disponível
        mode="min",
        patience=15,
        min_delta=1e-4,
        verbose=True
    )
    callbacks.append(early_stopping)
    
    # LR monitor
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="step")
    callbacks.append(lr_monitor)
    
    trainer = pl.Trainer(
        max_epochs=config.trainer.get('max_epochs', 100),
        gradient_clip_val=config.trainer.get('gradient_clip_val', 1.0),
        accumulate_grad_batches=config.trainer.get('accumulate_grad_batches', 4),
        precision=config.trainer.get('precision', 16),
        check_val_every_n_epoch=config.trainer.get('check_val_every_n_epoch', 5),
        log_every_n_steps=config.trainer.get('log_every_n_steps', 50),
        devices=config.trainer.get('devices', 1),
        accelerator=config.trainer.get('accelerator', 'auto'),
        callbacks=callbacks,
        logger=logger,
        deterministic=config.get('deterministic', True),
        benchmark=config.get('benchmark', False),
        num_sanity_val_steps=config.get('num_sanity_val_steps', 0),
    )
    
    return trainer


def create_model(config):
    """Cria modelo com cabeça Bayesiana"""
    logging.info("🧠 Criando modelo Moirai-MoE com cabeça Bayesiana...")
    
    # Por simplicidade, criar um modelo mock
    # Na implementação real, usar a fábrica de modelos do uni2ts
    
    class MockMoiraiBayesian(pl.LightningModule):
        def __init__(self, config):
            super().__init__()
            self.config = config
            self.save_hyperparameters()
            
            # Mock backbone (substituir por Moirai real)
            d_model = 512  # Seria extraído do modelo real
            self.backbone = torch.nn.Linear(100, d_model)  # Mock
            
            # Cabeça Bayesiana SOTA
            self.prediction_head = BayesianPredictionHead(
                d_model=d_model,
                prediction_length=config.model.prediction_length,
                mc_dropout_rate=config.model.prediction_head.mc_dropout_rate,
                num_mc_samples=config.model.prediction_head.num_mc_samples,
                use_variational_weights=config.model.prediction_head.use_variational_weights,
                use_temporal_attention=config.model.prediction_head.use_temporal_attention,
                student_t_df=config.model.prediction_head.student_t_df,
                confidence_levels=config.model.prediction_head.confidence_levels
            )
            
            # Loss Bayesiano SOTA
            self.criterion = BayesianELBOLoss(
                likelihood_weight=config.loss_func.likelihood_weight,
                kl_weight=config.loss_func.kl_weight,
                kl_annealing=config.loss_func.kl_annealing,
                kl_annealing_epochs=config.loss_func.kl_annealing_epochs,
                use_student_t=config.loss_func.use_student_t,
                student_t_df_min=config.loss_func.student_t_df_min,
            )
            
        def forward(self, x):
            # Mock forward
            batch_size = x.shape[0]
            seq_len = 2048
            d_model = 512
            
            # Simular saída do backbone
            reprs = torch.randn(batch_size, seq_len, d_model, device=self.device)
            
            # Cabeça Bayesiana
            return self.prediction_head(reprs, training=self.training)
        
        def training_step(self, batch, batch_idx):
            # Mock training step
            target = batch.get('future_target', torch.randn(16, 60, device=self.device))
            
            # Forward pass
            prediction_output = self(batch.get('target', torch.randn(16, 2048, 100, device=self.device)))
            
            # KL divergence
            kl_div = self.prediction_head.get_kl_divergence()
            
            # Loss
            loss_dict = self.criterion(prediction_output, target, kl_div)
            
            # Log métricas
            for key, value in loss_dict.items():
                self.log(f"train/{key}", value, prog_bar=(key == "loss"))
            
            return loss_dict["loss"]
        
        def validation_step(self, batch, batch_idx):
            # Mock validation step
            target = batch.get('future_target', torch.randn(16, 60, device=self.device))
            
            # Forward pass
            prediction_output = self(batch.get('target', torch.randn(16, 2048, 100, device=self.device)))
            
            # KL divergence
            kl_div = self.prediction_head.get_kl_divergence()
            
            # Loss
            loss_dict = self.criterion(prediction_output, target, kl_div)
            
            # Log métricas
            for key, value in loss_dict.items():
                self.log(f"val/{key}", value, prog_bar=(key == "loss"))
            
            return loss_dict["loss"]
        
        def configure_optimizers(self):
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=1e-4,
                weight_decay=1e-5,
                betas=[0.9, 0.999],
                eps=1e-8
            )
            
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=10,
                T_mult=2,
                eta_min=1e-6
            )
            
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch"
                }
            }
        
        def on_train_epoch_end(self):
            # Update KL annealing
            self.criterion.step_epoch()
    
    return MockMoiraiBayesian(config)


def create_dataloaders(config):
    """Cria dataloaders para treinamento"""
    logging.info("📊 Criando dataloaders...")
    
    # Mock dataloaders (substituir por implementação real)
    train_data = torch.utils.data.TensorDataset(
        torch.randn(1000, 2048, 100),  # Mock input
        torch.randn(1000, 60)          # Mock target
    )
    
    val_data = torch.utils.data.TensorDataset(
        torch.randn(200, 2048, 100),   # Mock input
        torch.randn(200, 60)           # Mock target
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=config.train_dataloader.batch_size,
        shuffle=config.train_dataloader.shuffle,
        num_workers=config.train_dataloader.num_workers,
        pin_memory=config.train_dataloader.pin_memory,
        drop_last=config.train_dataloader.drop_last
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=config.val_dataloader.batch_size,
        shuffle=config.val_dataloader.shuffle,
        num_workers=config.val_dataloader.num_workers,
        pin_memory=config.val_dataloader.pin_memory
    )
    
    return train_loader, val_loader


def main():
    parser = argparse.ArgumentParser(description="Fine-tuning SOTA Crypto com Moirai-MoE + Bayesiano")
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/crypto/finetune_bayesian_moe.yaml",
        help="Caminho para arquivo de configuração"
    )
    parser.add_argument(
        "--resume", 
        type=str, 
        default=None,
        help="Caminho para checkpoint para resumir treinamento"
    )
    parser.add_argument(
        "--debug", 
        action="store_true",
        help="Modo debug (fast_dev_run)"
    )
    
    args = parser.parse_args()
    
    # Carregar configuração
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"❌ Arquivo de configuração não encontrado: {config_path}")
        return 1
    
    config = OmegaConf.load(config_path)
    
    # Setup logging
    run_name = config.get('run_name', f"crypto_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    logger = setup_logging(run_name)
    
    logger.info("🚀 Iniciando fine-tuning SOTA Crypto com Moirai-MoE + Bayesiano")
    logger.info(f"📁 Configuração: {config_path}")
    logger.info(f"🏷️ Run name: {run_name}")
    
    try:
        # Validar configuração
        validate_config(config)
        logger.info("✅ Configuração validada")
        
        # Setup ambiente
        device_info = setup_environment()
        
        # Criar modelo
        model = create_model(config)
        logger.info("✅ Modelo criado com cabeça Bayesiana SOTA")
        
        # Criar dataloaders
        train_loader, val_loader = create_dataloaders(config)
        logger.info(f"✅ Dataloaders criados - Train: {len(train_loader)}, Val: {len(val_loader)}")
        
        # Criar trainer
        trainer = create_trainer(config)
        logger.info("✅ Trainer configurado")
        
        # Debug mode
        if args.debug:
            config.trainer.fast_dev_run = True
            logger.info("🐛 Modo debug ativado")
        
        # Iniciar treinamento
        logger.info("🎯 Iniciando treinamento...")
        
        if args.resume:
            logger.info(f"📂 Resumindo do checkpoint: {args.resume}")
            trainer.fit(model, train_loader, val_loader, ckpt_path=args.resume)
        else:
            trainer.fit(model, train_loader, val_loader)
        
        # Informações finais
        logger.info("✅ Treinamento concluído!")
        
        best_model_path = trainer.checkpoint_callback.best_model_path
        if best_model_path:
            logger.info(f"🏆 Melhor modelo salvo em: {best_model_path}")
        
        logger.info("📊 Métricas finais:")
        # Implementar extração de métricas do trainer
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Erro durante treinamento: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    exit(main())
