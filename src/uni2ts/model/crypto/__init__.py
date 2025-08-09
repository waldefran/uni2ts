"""
Módulo Crypto para uni2ts - Implementação SOTA BOOST.MD

Este módulo implementa todas as melhorias do BOOST.MD para trading de criptomoedas:
- Dataset unificado e anônimo  
- Features cíclicas explícitas
- Normalização por janela
- Cabeça Bayesiana SOTA
- Loss ELBO Bayesiana
- Compatibilidade Moirai-MoE
"""

from .bayesian_head import (
    BayesianPredictionHead,
    BayesianLinear, 
    BayesianPredictionOutput
)

__all__ = [
    "BayesianPredictionHead",
    "BayesianLinear", 
    "BayesianPredictionOutput"
]
