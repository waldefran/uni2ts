"""
Bayesian Prediction Head SOTA para Trading de Criptomoedas
Implementa quantificação de incerteza epistêmica + aleatória para séries temporais financeiras

CONFIRMADO (BOOST.MD): Mantido integralmente como estado da arte
- Camadas variacionais (BayesianLinear) para incerteza epistêmica
- MC Dropout para incerteza adicional do modelo
- Distribuição Student-T para heavy tails de crypto
- Decomposição rigorosa de incerteza: epistêmica vs aleatória
- Attention temporal para interpretabilidade
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, List
from torch.distributions import StudentT, Normal
from dataclasses import dataclass

from uni2ts.common.torch_util import as_dict


@dataclass
class BayesianPredictionOutput:
    """Saída estruturada da predição Bayesiana SOTA"""
    mean_prediction: torch.Tensor                    # [batch, pred_len] - predição média
    epistemic_uncertainty: torch.Tensor              # [batch, pred_len] - incerteza do modelo
    aleatoric_uncertainty: torch.Tensor              # [batch, pred_len] - incerteza dos dados
    total_uncertainty: torch.Tensor                  # [batch, pred_len] - incerteza total
    confidence_intervals: Dict[str, torch.Tensor]    # Intervalos de confiança (95%, 80%, etc.)
    mc_samples: torch.Tensor                         # [batch, pred_len, n_samples] - amostras MC
    attention_weights: Optional[torch.Tensor] = None # [batch, seq_len] - pesos atenção temporal
    student_t_params: Optional[Dict[str, torch.Tensor]] = None  # Parâmetros da distribuição


class BayesianLinear(nn.Module):
    """
    Camada Linear Variacional SOTA para incerteza epistêmica
    Implementa reparameterization trick para pesos Bayesianos
    """
    def __init__(self, in_features: int, out_features: int, prior_std: float = 1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_std = prior_std
        
        # Parâmetros variacionais para pesos
        self.weight_mu = nn.Parameter(torch.zeros(out_features, in_features))
        self.weight_logsigma = nn.Parameter(torch.full((out_features, in_features), -3.0))
        
        # Parâmetros variacionais para bias
        self.bias_mu = nn.Parameter(torch.zeros(out_features))
        self.bias_logsigma = nn.Parameter(torch.full((out_features,), -3.0))
        
        # Inicialização Xavier para médias
        nn.init.xavier_uniform_(self.weight_mu)
        nn.init.zeros_(self.bias_mu)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass com reparameterization trick"""
        # Pesos variacionais
        weight_sigma = torch.exp(self.weight_logsigma)
        weight_eps = torch.randn_like(self.weight_mu)
        weight = self.weight_mu + weight_sigma * weight_eps
        
        # Bias variacional
        bias_sigma = torch.exp(self.bias_logsigma)
        bias_eps = torch.randn_like(self.bias_mu)
        bias = self.bias_mu + bias_sigma * bias_eps
        
        return F.linear(x, weight, bias)
    
    def kl_divergence(self) -> torch.Tensor:
        """Calcula divergência KL para regularização ELBO"""
        # KL para pesos
        weight_var = torch.exp(2 * self.weight_logsigma)
        weight_kl = 0.5 * torch.sum(
            self.weight_mu**2 / self.prior_std**2 + 
            weight_var / self.prior_std**2 - 
            2 * self.weight_logsigma + 
            2 * np.log(self.prior_std) - 1
        )
        
        # KL para bias
        bias_var = torch.exp(2 * self.bias_logsigma)
        bias_kl = 0.5 * torch.sum(
            self.bias_mu**2 / self.prior_std**2 + 
            bias_var / self.prior_std**2 - 
            2 * self.bias_logsigma + 
            2 * np.log(self.prior_std) - 1
        )
        
        return weight_kl + bias_kl


class BayesianPredictionHead(nn.Module):
    """
    Cabeça de Predição Bayesiana SOTA para Trading de Criptomoedas
    
    Estado da arte implementado:
    - Quantificação de incerteza epistêmica (modelo) + aleatória (dados)
    - MC Dropout para robustez adicional
    - Distribuição Student-T para heavy tails
    - Attention temporal para interpretabilidade
    - Decomposição rigorosa de incerteza
    """
    
    def __init__(
        self,
        d_model: int,
        prediction_length: int,
        mc_dropout_rate: float = 0.15,
        num_mc_samples: int = 100,
        use_variational_weights: bool = True,
        use_temporal_attention: bool = True,
        student_t_df: float = 4.0,  # Graus de liberdade para heavy tails
        confidence_levels: List[float] = [0.8, 0.9, 0.95, 0.99]
    ):
        super().__init__()
        self.d_model = d_model
        self.prediction_length = prediction_length
        self.mc_dropout_rate = mc_dropout_rate
        self.num_mc_samples = num_mc_samples
        self.use_variational_weights = use_variational_weights
        self.use_temporal_attention = use_temporal_attention
        self.student_t_df = student_t_df
        self.confidence_levels = confidence_levels
        
        # Attention temporal para focar em padrões importantes
        if use_temporal_attention:
            self.temporal_attention = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=8,
                dropout=0.1,
                batch_first=True
            )
            self.attention_norm = nn.LayerNorm(d_model)
        
        # Camadas de processamento
        if use_variational_weights:
            self.variational_linear1 = BayesianLinear(d_model, d_model)
            self.variational_linear2 = BayesianLinear(d_model, d_model // 2)
        else:
            self.variational_linear1 = nn.Linear(d_model, d_model)
            self.variational_linear2 = nn.Linear(d_model, d_model // 2)
        
        # MC Dropout para incerteza epistêmica adicional
        self.mc_dropout = nn.Dropout(mc_dropout_rate)
        self.feature_dropout = nn.Dropout(0.1)
        
        # Normalização
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model // 2)
        
        # Projeção final para parâmetros da distribuição Student-T
        # Student-T tem 3 parâmetros: loc (média), scale (escala), df (graus liberdade)
        self.param_projection = nn.Linear(d_model // 2, prediction_length * 3)
        
        # Ativações não-lineares (SiLU é estado da arte)
        self.activation = nn.SiLU()
        
    def forward(
        self, 
        reprs: torch.Tensor,  # [batch, seq_len, d_model]
        training: bool = True
    ) -> BayesianPredictionOutput:
        """
        Forward pass com amostragem Monte Carlo para quantificação de incerteza
        
        Args:
            reprs: Representações do encoder [batch, seq_len, d_model]
            training: Se está em modo de treinamento
            
        Returns:
            BayesianPredictionOutput com decomposição completa de incerteza
        """
        batch_size, seq_len, d_model = reprs.shape
        
        # Durante treinamento: uma passada + dropout
        # Durante inferência: múltiplas passadas MC para quantificar incerteza
        num_samples = 1 if training else self.num_mc_samples
        
        mc_outputs = []
        attention_weights_list = []
        
        for _ in range(num_samples):
            # 1. Attention temporal (se habilitado)
            if self.use_temporal_attention:
                attended_reprs, attention_weights = self.temporal_attention(
                    reprs, reprs, reprs
                )
                attended_reprs = self.attention_norm(attended_reprs + reprs)
                attention_weights_list.append(attention_weights.mean(dim=1))  # [batch, seq_len]
            else:
                attended_reprs = reprs
                attention_weights = None
            
            # 2. Agregação temporal (média ponderada ou pooling)
            if self.use_temporal_attention:
                # Usar attention weights para pooling ponderado
                attn_pooled = torch.sum(attended_reprs * attention_weights.unsqueeze(-1), dim=1)
            else:
                # Pooling simples
                attn_pooled = torch.mean(attended_reprs, dim=1)  # [batch, d_model]
            
            # 3. Camadas variacionais + MC Dropout
            h = self.layer_norm1(attn_pooled)
            h = self.variational_linear1(h)
            h = self.activation(h)
            h = self.mc_dropout(h) if training or num_samples > 1 else h
            
            h = self.layer_norm2(h)
            h = self.variational_linear2(h)
            h = self.activation(h)
            h = self.feature_dropout(h)
            
            # 4. Projeção para parâmetros da distribuição
            params = self.param_projection(h)  # [batch, pred_len * 3]
            params = params.view(batch_size, self.prediction_length, 3)
            
            mc_outputs.append(params)
        
        # Empilhar todas as amostras MC
        mc_outputs = torch.stack(mc_outputs, dim=-1)  # [batch, pred_len, 3, n_samples]
        
        # Agregar attention weights se disponível
        if attention_weights_list:
            attention_weights = torch.stack(attention_weights_list, dim=0).mean(dim=0)
        else:
            attention_weights = None
        
        # Processar saídas e calcular incertezas
        return self._process_mc_outputs(mc_outputs, attention_weights)
    
    def _process_mc_outputs(
        self, 
        mc_outputs: torch.Tensor,  # [batch, pred_len, 3, n_samples]
        attention_weights: Optional[torch.Tensor]
    ) -> BayesianPredictionOutput:
        """Processa amostras MC e calcula decomposição de incerteza"""
        
        # Extrair parâmetros da distribuição Student-T
        loc_samples = mc_outputs[:, :, 0, :]      # [batch, pred_len, n_samples]
        scale_samples = F.softplus(mc_outputs[:, :, 1, :]) + 1e-6  # Garantir positividade
        df_samples = F.softplus(mc_outputs[:, :, 2, :]) + 2.0     # df > 2 para variância finita
        
        # 1. Predição média (consenso do ensemble)
        mean_prediction = torch.mean(loc_samples, dim=-1)  # [batch, pred_len]
        
        # 2. Incerteza epistêmica (variação do modelo)
        epistemic_uncertainty = torch.var(loc_samples, dim=-1)  # [batch, pred_len]
        
        # 3. Incerteza aleatória (média das incertezas individuais)
        # Para Student-T: var = scale^2 * df/(df-2) quando df > 2
        individual_variances = scale_samples**2 * df_samples / (df_samples - 2)
        aleatoric_uncertainty = torch.mean(individual_variances, dim=-1)  # [batch, pred_len]
        
        # 4. Incerteza total
        total_uncertainty = epistemic_uncertainty + aleatoric_uncertainty
        
        # 5. Intervalos de confiança usando distribuição Student-T
        confidence_intervals = {}
        mean_scale = torch.mean(scale_samples, dim=-1)
        mean_df = torch.mean(df_samples, dim=-1)
        
        for conf_level in self.confidence_levels:
            # Quantis da distribuição Student-T
            alpha = 1 - conf_level
            # Aproximação para quantis (pode ser melhorada com scipy.stats)
            z_score = torch.ones_like(mean_prediction) * torch.tensor(
                {0.8: 1.282, 0.9: 1.645, 0.95: 1.96, 0.99: 2.576}.get(conf_level, 1.96)
            )
            
            margin = z_score * mean_scale * torch.sqrt(total_uncertainty)
            confidence_intervals[f"{int(conf_level*100)}%"] = {
                "lower": mean_prediction - margin,
                "upper": mean_prediction + margin
            }
        
        # 6. Parâmetros Student-T para distribuição final
        student_t_params = {
            "loc": torch.mean(loc_samples, dim=-1),
            "scale": torch.mean(scale_samples, dim=-1),
            "df": torch.mean(df_samples, dim=-1)
        }
        
        return BayesianPredictionOutput(
            mean_prediction=mean_prediction,
            epistemic_uncertainty=epistemic_uncertainty,
            aleatoric_uncertainty=aleatoric_uncertainty,
            total_uncertainty=total_uncertainty,
            confidence_intervals=confidence_intervals,
            mc_samples=loc_samples,  # Guardar amostras para análises posteriores
            attention_weights=attention_weights,
            student_t_params=student_t_params
        )
    
    def get_kl_divergence(self) -> torch.Tensor:
        """Calcula divergência KL total para ELBO loss"""
        if not self.use_variational_weights:
            return torch.tensor(0.0)
        
        kl_total = torch.tensor(0.0)
        
        if isinstance(self.variational_linear1, BayesianLinear):
            kl_total += self.variational_linear1.kl_divergence()
        if isinstance(self.variational_linear2, BayesianLinear):
            kl_total += self.variational_linear2.kl_divergence()
            
        return kl_total
    
    def sample_predictions(
        self, 
        reprs: torch.Tensor, 
        num_samples: int = 100
    ) -> torch.Tensor:
        """
        Gera amostras diretas da distribuição preditiva
        Útil para análises de risco e backtesting
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(reprs, training=False)
            
            # Criar distribuição Student-T e amostrar
            dist = StudentT(
                df=output.student_t_params["df"],
                loc=output.student_t_params["loc"],
                scale=output.student_t_params["scale"]
            )
            
            samples = dist.sample((num_samples,))  # [num_samples, batch, pred_len]
            return samples.permute(1, 2, 0)  # [batch, pred_len, num_samples]
