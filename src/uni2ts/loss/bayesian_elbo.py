"""
Bayesian ELBO Loss SOTA para Trading de Criptomoedas
Implementa Evidence Lower Bound com regularização KL para redes variacionais

CONFIRMADO (BOOST.MD): Mantida integralmente como estado da arte
- Matematicamente mais rigorosa que NLL pura
- Inclui regularização KL essencial para redes variacionais
- Estabiliza treinamento de pesos variacionais
- Adaptada para distribuições Student-T (heavy tails crypto)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import StudentT, Normal
from typing import Dict, Optional, Union
import numpy as np

from uni2ts.model.crypto.bayesian_head import BayesianPredictionOutput


class BayesianELBOLoss(nn.Module):
    """
    Evidence Lower Bound Loss SOTA para modelos Bayesianos
    
    ELBO = E[log p(y|x)] - β * KL[q(θ)|p(θ)]
    
    Onde:
    - E[log p(y|x)]: Likelihood esperada (predição)
    - KL[q(θ)|p(θ)]: Divergência KL entre posterior e prior (regularização)
    - β: Peso da regularização KL (annealing schedule)
    """
    
    def __init__(
        self,
        likelihood_weight: float = 1.0,
        kl_weight: float = 1e-4,
        kl_annealing: bool = True,
        kl_annealing_epochs: int = 100,
        use_student_t: bool = True,
        student_t_df_min: float = 2.1,  # Para garantir variância finita
        reduction: str = "mean"
    ):
        super().__init__()
        self.likelihood_weight = likelihood_weight
        self.initial_kl_weight = kl_weight
        self.kl_weight = kl_weight
        self.kl_annealing = kl_annealing
        self.kl_annealing_epochs = kl_annealing_epochs
        self.use_student_t = use_student_t
        self.student_t_df_min = student_t_df_min
        self.reduction = reduction
        
        # Contador de épocas para KL annealing
        self.register_buffer('epoch_counter', torch.tensor(0.0))
        
    def forward(
        self,
        prediction_output: BayesianPredictionOutput,
        target: torch.Tensor,  # [batch, pred_len]
        kl_divergence: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Calcula ELBO loss com decomposição detalhada
        
        Args:
            prediction_output: Saída da BayesianPredictionHead
            target: Valores verdadeiros
            kl_divergence: Divergência KL das camadas variacionais
            mask: Máscara para valores válidos (opcional)
            
        Returns:
            Dict com loss total e componentes individuais
        """
        batch_size = target.shape[0]
        
        # 1. Likelihood Term: -log p(y|x,θ)
        likelihood_loss = self._compute_likelihood_loss(prediction_output, target, mask)
        
        # 2. KL Divergence Term: KL[q(θ)|p(θ)]
        kl_loss = self._compute_kl_loss(kl_divergence, batch_size)
        
        # 3. Update KL weight (annealing schedule)
        current_kl_weight = self._update_kl_weight()
        
        # 4. ELBO Total
        elbo_loss = (
            self.likelihood_weight * likelihood_loss + 
            current_kl_weight * kl_loss
        )
        
        # 5. Métricas adicionais para monitoramento
        uncertainty_metrics = self._compute_uncertainty_metrics(prediction_output, target)
        
        return {
            # Loss principal
            "loss": elbo_loss,
            
            # Componentes ELBO
            "likelihood_loss": likelihood_loss,
            "kl_loss": kl_loss,
            "kl_weight": current_kl_weight,
            
            # Métricas de monitoramento
            "mean_epistemic_uncertainty": uncertainty_metrics["mean_epistemic"],
            "mean_aleatoric_uncertainty": uncertainty_metrics["mean_aleatoric"],
            "uncertainty_ratio": uncertainty_metrics["uncertainty_ratio"],
            "prediction_sharpness": uncertainty_metrics["sharpness"],
            
            # Métricas de calibração
            "mean_prediction_error": uncertainty_metrics["prediction_error"],
            "calibration_score": uncertainty_metrics["calibration_score"]
        }
    
    def _compute_likelihood_loss(
        self,
        prediction_output: BayesianPredictionOutput,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Computa loss de likelihood usando distribuição Student-T ou Normal"""
        
        if self.use_student_t and prediction_output.student_t_params is not None:
            # Usar distribuição Student-T para heavy tails
            loc = prediction_output.student_t_params["loc"]
            scale = prediction_output.student_t_params["scale"]
            df = torch.clamp(prediction_output.student_t_params["df"], min=self.student_t_df_min)
            
            # Criar distribuição Student-T
            dist = StudentT(df=df, loc=loc, scale=scale)
            
        else:
            # Fallback para distribuição Normal
            loc = prediction_output.mean_prediction
            scale = torch.sqrt(prediction_output.total_uncertainty) + 1e-6
            dist = Normal(loc=loc, scale=scale)
        
        # Negative log-likelihood
        nll = -dist.log_prob(target)
        
        # Aplicar máscara se fornecida
        if mask is not None:
            nll = nll * mask
            
        # Redução
        if self.reduction == "mean":
            if mask is not None:
                return nll.sum() / mask.sum()
            else:
                return nll.mean()
        elif self.reduction == "sum":
            return nll.sum()
        else:
            return nll
    
    def _compute_kl_loss(self, kl_divergence: torch.Tensor, batch_size: int) -> torch.Tensor:
        """Computa perda de divergência KL normalizada pelo batch size"""
        # Normalizar KL pelo tamanho do batch (importante para estabilidade)
        return kl_divergence / batch_size
    
    def _update_kl_weight(self) -> float:
        """Update do peso KL com annealing schedule"""
        if not self.kl_annealing:
            return self.kl_weight
        
        # Linear annealing: começa em 0 e cresce até initial_kl_weight
        progress = min(1.0, self.epoch_counter.item() / self.kl_annealing_epochs)
        current_weight = self.initial_kl_weight * progress
        
        return current_weight
    
    def _compute_uncertainty_metrics(
        self,
        prediction_output: BayesianPredictionOutput,
        target: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Computa métricas de incerteza para monitoramento"""
        
        # Médias das incertezas
        mean_epistemic = torch.mean(prediction_output.epistemic_uncertainty)
        mean_aleatoric = torch.mean(prediction_output.aleatoric_uncertainty)
        
        # Ratio epistêmica/aleatória (indica qualidade do modelo)
        total_uncertainty = mean_epistemic + mean_aleatoric
        uncertainty_ratio = mean_epistemic / (total_uncertainty + 1e-8)
        
        # Sharpness: quão "afiadas" são as predições (menor incerteza = melhor)
        sharpness = 1.0 / (torch.mean(prediction_output.total_uncertainty) + 1e-8)
        
        # Erro de predição
        prediction_error = torch.mean(torch.abs(prediction_output.mean_prediction - target))
        
        # Score de calibração simples
        # Idealmente, incerteza deveria correlacionar com erro
        errors = torch.abs(prediction_output.mean_prediction - target).flatten()
        uncertainties = torch.sqrt(prediction_output.total_uncertainty).flatten()
        
        # Correlação entre erro e incerteza (boa calibração ~= correlação alta)
        if len(errors) > 1:
            errors_norm = (errors - errors.mean()) / (errors.std() + 1e-8)
            uncertainties_norm = (uncertainties - uncertainties.mean()) / (uncertainties.std() + 1e-8)
            calibration_score = torch.mean(errors_norm * uncertainties_norm)
        else:
            calibration_score = torch.tensor(0.0)
        
        return {
            "mean_epistemic": mean_epistemic,
            "mean_aleatoric": mean_aleatoric,
            "uncertainty_ratio": uncertainty_ratio,
            "sharpness": sharpness,
            "prediction_error": prediction_error,
            "calibration_score": calibration_score
        }
    
    def step_epoch(self):
        """Chama ao final de cada época para atualizar annealing"""
        self.epoch_counter += 1
    
    def get_kl_weight(self) -> float:
        """Retorna o peso KL atual"""
        return self._update_kl_weight()


class BayesianMetricsTracker:
    """
    Tracker para métricas Bayesianas ao longo do treinamento
    Útil para monitorar qualidade da quantificação de incerteza
    """
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.reset()
    
    def reset(self):
        """Reset das métricas"""
        self.epistemic_history = []
        self.aleatoric_history = []
        self.calibration_history = []
        self.sharpness_history = []
        
    def update(self, loss_dict: Dict[str, torch.Tensor]):
        """Update com saída da BayesianELBOLoss"""
        self.epistemic_history.append(loss_dict["mean_epistemic_uncertainty"].item())
        self.aleatoric_history.append(loss_dict["mean_aleatoric_uncertainty"].item())
        self.calibration_history.append(loss_dict["calibration_score"].item())
        self.sharpness_history.append(loss_dict["prediction_sharpness"].item())
        
        # Manter apenas últimas observações
        if len(self.epistemic_history) > self.window_size:
            self.epistemic_history.pop(0)
            self.aleatoric_history.pop(0)
            self.calibration_history.pop(0)
            self.sharpness_history.pop(0)
    
    def get_summary(self) -> Dict[str, float]:
        """Sumário das métricas recentes"""
        if not self.epistemic_history:
            return {}
        
        return {
            "avg_epistemic_uncertainty": np.mean(self.epistemic_history),
            "avg_aleatoric_uncertainty": np.mean(self.aleatoric_history),
            "avg_calibration_score": np.mean(self.calibration_history),
            "avg_sharpness": np.mean(self.sharpness_history),
            "uncertainty_stability": np.std(self.epistemic_history),
            "calibration_trend": np.mean(self.calibration_history[-10:]) - np.mean(self.calibration_history[:10]) if len(self.calibration_history) >= 20 else 0.0
        }


def create_bayesian_elbo_loss(
    likelihood_weight: float = 1.0,
    kl_weight: float = 1e-4,
    **kwargs
) -> BayesianELBOLoss:
    """
    Factory function para criar BayesianELBOLoss com configuração SOTA
    
    Args:
        likelihood_weight: Peso do termo likelihood
        kl_weight: Peso da regularização KL
        **kwargs: Outros parâmetros para BayesianELBOLoss
    
    Returns:
        Instância configurada de BayesianELBOLoss
    """
    return BayesianELBOLoss(
        likelihood_weight=likelihood_weight,
        kl_weight=kl_weight,
        kl_annealing=True,
        kl_annealing_epochs=50,  # Annealing mais rápido para convergência
        use_student_t=True,      # Heavy tails para crypto
        **kwargs
    )
