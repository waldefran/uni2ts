"""
Bayesian Uncertainty Monitor Callback SOTA
Monitora e registra métricas de incerteza durante fine-tuning Bayesiano

SOTA implementado:
- Monitoramento de incerteza epistêmica vs aleatória
- Detecção de regimes de alta/baixa incerteza
- Alertas para degradação de calibração
- Análise de attention weights para interpretabilidade
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from typing import Dict, Optional, List
import numpy as np
from pathlib import Path

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from uni2ts.model.crypto.bayesian_head import BayesianPredictionOutput


class BayesianUncertaintyMonitor(Callback):
    """
    Callback SOTA para monitorar incerteza em modelos Bayesianos durante treinamento
    
    Features implementadas:
    - Decomposição de incerteza (epistêmica vs aleatória)
    - Monitoramento de calibração contínua
    - Alertas para degradação de incerteza
    - Visualização de attention weights
    - Tracking de regimes de volatilidade
    """
    
    def __init__(
        self,
        log_attention_weights: bool = True,
        uncertainty_threshold: float = 0.05,
        log_interval: int = 100,
        save_plots: bool = True,
        plot_dir: str = "./uncertainty_plots"
    ):
        super().__init__()
        self.log_attention_weights = log_attention_weights
        self.uncertainty_threshold = uncertainty_threshold
        self.log_interval = log_interval
        self.save_plots = save_plots
        self.plot_dir = Path(plot_dir)
        
        # Criar diretório de plots se necessário
        if self.save_plots:
            self.plot_dir.mkdir(parents=True, exist_ok=True)
        
        # Histórico de métricas
        self.uncertainty_history = {
            "epistemic": [],
            "aleatoric": [],
            "total": [],
            "ratio": []  # epistemic/total
        }
        self.attention_history = []
        self.step_count = 0
        
    def on_validation_batch_end(
        self, 
        trainer: pl.Trainer, 
        pl_module: pl.LightningModule, 
        outputs: any, 
        batch: Dict[str, torch.Tensor], 
        batch_idx: int, 
        dataloader_idx: int = 0
    ):
        """Processa cada batch de validação para extrair métricas de incerteza"""
        
        # Verificar se o modelo tem cabeça Bayesiana
        if not hasattr(pl_module, 'prediction_head'):
            return
        
        # Buscar prediction_output no dicionário retornado pelo validation_step
        prediction_output = None
        if isinstance(outputs, dict) and 'prediction_output' in outputs:
            prediction_output = outputs['prediction_output']
        elif hasattr(outputs, 'epistemic_uncertainty'):
            prediction_output = outputs
        
        if prediction_output is not None:
            with torch.no_grad():
                self._process_bayesian_output(prediction_output, trainer.global_step, trainer)
    
    def _process_bayesian_output(self, output: BayesianPredictionOutput, global_step: int, trainer: pl.Trainer = None):
        """Processa saída Bayesiana e registra métricas"""
        
        # Calcular métricas de incerteza
        epistemic_mean = output.epistemic_uncertainty.mean().item()
        aleatoric_mean = output.aleatoric_uncertainty.mean().item()
        total_mean = output.total_uncertainty.mean().item()
        
        # Ratio epistêmica/total (indica confiança do modelo)
        epistemic_ratio = epistemic_mean / (total_mean + 1e-8)
        
        # Armazenar histórico
        self.uncertainty_history["epistemic"].append(epistemic_mean)
        self.uncertainty_history["aleatoric"].append(aleatoric_mean)
        self.uncertainty_history["total"].append(total_mean)
        self.uncertainty_history["ratio"].append(epistemic_ratio)
        
        # Log métricas
        metrics = {
            "uncertainty/epistemic_mean": epistemic_mean,
            "uncertainty/aleatoric_mean": aleatoric_mean,
            "uncertainty/total_mean": total_mean,
            "uncertainty/epistemic_ratio": epistemic_ratio,
            "uncertainty/model_confidence": 1.0 - epistemic_ratio,
        }
        
        # Detectar regime de alta incerteza
        if total_mean > self.uncertainty_threshold:
            metrics["uncertainty/high_uncertainty_alert"] = 1.0
        
        # Log attention weights se disponível
        if self.log_attention_weights and output.attention_weights is not None:
            self._process_attention_weights(output.attention_weights, metrics)
        
        # Registrar métricas - CORREÇÃO: usar trainer do argumento
        if trainer is not None and hasattr(trainer, 'logger') and trainer.logger is not None:
            trainer.logger.log_metrics(metrics, step=global_step)
        
        self.step_count += 1
        
        # Criar plots periodicamente
        if self.save_plots and self.step_count % self.log_interval == 0:
            self._create_uncertainty_plots(global_step)
    
    def _process_attention_weights(self, attention_weights: torch.Tensor, metrics: Dict):
        """Processa e registra pesos de attention"""
        
        # Calcular entropia da attention (dispersão)
        attention_probs = torch.softmax(attention_weights, dim=-1)
        attention_entropy = -torch.sum(attention_probs * torch.log(attention_probs + 1e-8), dim=-1)
        entropy_mean = attention_entropy.mean().item()
        
        # Attention concentration (inverso da entropia)
        max_entropy = np.log(attention_weights.shape[-1])
        attention_concentration = 1.0 - (entropy_mean / max_entropy)
        
        metrics.update({
            "attention/entropy_mean": entropy_mean,
            "attention/concentration": attention_concentration,
            "attention/focus_score": attention_concentration  # Alias para interpretabilidade
        })
        
        # Armazenar para análise posterior
        self.attention_history.append(entropy_mean)
    
    def _create_uncertainty_plots(self, global_step: int):
        """Cria visualizações de incerteza - CORREÇÃO: matplotlib local"""
        
        if len(self.uncertainty_history["total"]) < 10:
            return  # Poucos dados para plotar
        
        # Importar matplotlib localmente para evitar efeitos globais
        import matplotlib
        matplotlib.use('Agg')  # Backend não-GUI apenas para esta função
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Plot 1: Decomposição de incerteza
        axes[0, 0].plot(self.uncertainty_history["epistemic"], label="Epistêmica", alpha=0.8)
        axes[0, 0].plot(self.uncertainty_history["aleatoric"], label="Aleatória", alpha=0.8)
        axes[0, 0].plot(self.uncertainty_history["total"], label="Total", alpha=0.8)
        axes[0, 0].set_title("Decomposição de Incerteza")
        axes[0, 0].set_ylabel("Incerteza")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Ratio epistêmica/total (confiança do modelo)
        axes[0, 1].plot(self.uncertainty_history["ratio"], color="red", alpha=0.8)
        axes[0, 1].axhline(y=0.5, color="black", linestyle="--", alpha=0.5)
        axes[0, 1].set_title("Confiança do Modelo (1 - Epistêmica/Total)")
        axes[0, 1].set_ylabel("Ratio Epistêmica")
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Histograma de incerteza total
        recent_uncertainty = self.uncertainty_history["total"][-100:]  # Últimos 100 steps
        axes[1, 0].hist(recent_uncertainty, bins=20, alpha=0.7, color="blue")
        axes[1, 0].axvline(x=self.uncertainty_threshold, color="red", linestyle="--", 
                          label=f"Threshold ({self.uncertainty_threshold})")
        axes[1, 0].set_title("Distribuição de Incerteza (últimos 100 steps)")
        axes[1, 0].set_xlabel("Incerteza Total")
        axes[1, 0].legend()
        
        # Plot 4: Attention entropy (se disponível)
        if self.attention_history:
            axes[1, 1].plot(self.attention_history, color="green", alpha=0.8)
            axes[1, 1].set_title("Entropia de Attention")
            axes[1, 1].set_ylabel("Entropia")
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, "Attention weights\nnão disponíveis", 
                           ha="center", va="center", transform=axes[1, 1].transAxes)
        
        plt.tight_layout()
        
        # Salvar plot
        plot_path = self.plot_dir / f"uncertainty_step_{global_step}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        
        # Log para WandB se disponível
        if WANDB_AVAILABLE:
            try:
                wandb.log({"uncertainty_analysis": wandb.Image(str(plot_path))}, step=global_step)
            except:
                pass  # WandB não disponível
    
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Resumo das métricas ao final de cada época"""
        
        if len(self.uncertainty_history["total"]) == 0:
            return
        
        # Calcular estatísticas da época
        recent_window = min(100, len(self.uncertainty_history["total"]))
        recent_uncertainty = self.uncertainty_history["total"][-recent_window:]
        recent_ratio = self.uncertainty_history["ratio"][-recent_window:]
        
        epoch_metrics = {
            "uncertainty_epoch/mean_total": np.mean(recent_uncertainty),
            "uncertainty_epoch/std_total": np.std(recent_uncertainty),
            "uncertainty_epoch/mean_confidence": 1.0 - np.mean(recent_ratio),
            "uncertainty_epoch/high_uncertainty_pct": np.mean([u > self.uncertainty_threshold for u in recent_uncertainty]) * 100
        }
        
        # Log métricas da época usando o logger do trainer
        if trainer.logger:
            trainer.logger.log_metrics(epoch_metrics, step=trainer.global_step)
    
    def get_uncertainty_summary(self) -> Dict:
        """Retorna resumo das métricas de incerteza"""
        
        if len(self.uncertainty_history["total"]) == 0:
            return {}
        
        return {
            "total_steps": len(self.uncertainty_history["total"]),
            "mean_epistemic": np.mean(self.uncertainty_history["epistemic"]),
            "mean_aleatoric": np.mean(self.uncertainty_history["aleatoric"]),
            "mean_total": np.mean(self.uncertainty_history["total"]),
            "mean_confidence": 1.0 - np.mean(self.uncertainty_history["ratio"]),
            "high_uncertainty_episodes": np.sum([u > self.uncertainty_threshold for u in self.uncertainty_history["total"]]),
            "calibration_trend": "improving" if len(self.uncertainty_history["ratio"]) > 10 and 
                               np.polyfit(range(len(self.uncertainty_history["ratio"])), self.uncertainty_history["ratio"], 1)[0] < 0 
                               else "stable/degrading"
        }
