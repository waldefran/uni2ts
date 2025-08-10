"""
Métricas Bayesianas SOTA para Avaliação de Modelos Crypto
Implementa métricas específicas para calibração e qualidade de incerteza

SOTA implementado:
- Expected Calibration Error (ECE)
- Prediction Interval Coverage (PIT)
- Reliability Diagram
- Uncertainty Decomposition Metrics
- Sharpness Score
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy import stats
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt

from uni2ts.model.crypto.bayesian_head import BayesianPredictionOutput


@dataclass
class BayesianMetrics:
    """Container para métricas Bayesianas"""
    ece: float  # Expected Calibration Error
    coverage: Dict[str, float]  # Coverage por nível de confiança
    pit_histogram: np.ndarray  # Probability Integral Transform
    sharpness: float  # Média da largura dos intervalos
    epistemic_ratio: float  # Ratio incerteza epistêmica/total
    reliability_score: float  # Score de confiabilidade 0-1


class BayesianCalibration:
    """
    Métrica SOTA para calibração Bayesiana
    
    Calcula Expected Calibration Error (ECE) que mede o quão bem
    calibradas estão as predições probabilísticas
    """
    
    def __init__(self, n_bins: int = 15):
        self.n_bins = n_bins
        
    def __call__(
        self, 
        predictions: BayesianPredictionOutput, 
        targets: torch.Tensor
    ) -> float:
        """
        Calcula ECE (Expected Calibration Error)
        
        Args:
            predictions: Saída Bayesiana com intervalos de confiança
            targets: Valores reais
            
        Returns:
            ECE score (menor = melhor calibração)
        """
        
        ece_total = 0.0
        confidence_levels = [0.8, 0.9, 0.95, 0.99]
        
        for conf_level in confidence_levels:
            conf_key = f"{int(conf_level*100)}%"
            if conf_key not in predictions.confidence_intervals:
                continue
                
            lower = predictions.confidence_intervals[conf_key]["lower"]
            upper = predictions.confidence_intervals[conf_key]["upper"]
            
            # Verificar se targets estão dentro dos intervalos
            in_interval = (targets >= lower) & (targets <= upper)
            
            # Calcular ECE para este nível de confiança
            ece = self._calculate_ece_single_level(
                in_interval.float(), 
                torch.ones_like(in_interval.float()) * conf_level
            )
            ece_total += ece
            
        return ece_total / len(confidence_levels)
    
    def _calculate_ece_single_level(
        self, 
        in_interval: torch.Tensor, 
        confidence: torch.Tensor
    ) -> float:
        """Calcula ECE para um nível de confiança específico"""
        
        bin_boundaries = torch.linspace(0, 1, self.n_bins + 1)
        ece = 0.0
        
        for i in range(self.n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            
            # Amostras neste bin
            bin_mask = (confidence > bin_lower) & (confidence <= bin_upper)
            
            if bin_mask.sum() > 0:
                # Confiança média no bin
                bin_confidence = confidence[bin_mask].mean()
                # Accuracy real no bin
                bin_accuracy = in_interval[bin_mask].mean()
                # Peso do bin
                bin_weight = bin_mask.sum().float() / len(confidence)
                
                # Contribuição para ECE
                ece += bin_weight * torch.abs(bin_confidence - bin_accuracy)
        
        return ece.item()


class UncertaintyDecomposition:
    """
    Métrica SOTA para decomposição de incerteza
    
    Analisa a qualidade da separação entre incerteza epistêmica e aleatória
    """
    
    def __call__(
        self, 
        predictions: BayesianPredictionOutput
    ) -> Dict[str, float]:
        """
        Analisa decomposição de incerteza
        
        Returns:
            Dict com métricas de decomposição
        """
        
        epistemic = predictions.epistemic_uncertainty
        aleatoric = predictions.aleatoric_uncertainty
        total = predictions.total_uncertainty
        
        # Ratio epistêmica/total (indica se modelo sabe quando não sabe)
        epistemic_ratio = (epistemic / (total + 1e-8)).mean()
        
        # Variabilidade da incerteza (diversidade de predições)
        uncertainty_variance = torch.var(total)
        
        # Correlação entre tipos de incerteza (idealmente baixa)
        correlation = torch.corrcoef(torch.stack([
            epistemic.flatten(), 
            aleatoric.flatten()
        ]))[0, 1]
        
        return {
            "epistemic_ratio": epistemic_ratio.item(),
            "aleatoric_ratio": (1.0 - epistemic_ratio).item(),
            "uncertainty_variance": uncertainty_variance.item(),
            "epistemic_aleatoric_correlation": correlation.item(),
            "uncertainty_spread": torch.std(total).item(),
            "confidence_score": (1.0 - epistemic_ratio).item()  # Confiança = menos epistêmica
        }


class PredictionSharpness:
    """
    Métrica SOTA para sharpness (precisão) dos intervalos de predição
    
    Intervalos mais estreitos = predições mais precisas (mas deve balancear com coverage)
    """
    
    def __call__(
        self, 
        predictions: BayesianPredictionOutput
    ) -> Dict[str, float]:
        """
        Calcula sharpness para diferentes níveis de confiança
        
        Returns:
            Dict com scores de sharpness (menor = mais preciso)
        """
        
        sharpness_scores = {}
        
        for conf_key, interval in predictions.confidence_intervals.items():
            lower = interval["lower"]
            upper = interval["upper"]
            
            # Largura média dos intervalos
            interval_width = (upper - lower).mean()
            
            # Normalizar pela magnitude dos valores (para comparabilidade)
            magnitude = torch.abs(predictions.mean_prediction).mean()
            normalized_width = interval_width / (magnitude + 1e-8)
            
            sharpness_scores[f"sharpness_{conf_key}"] = interval_width.item()
            sharpness_scores[f"normalized_sharpness_{conf_key}"] = normalized_width.item()
        
        # Sharpness geral (média de todos os níveis)
        all_sharpness = [v for k, v in sharpness_scores.items() if k.startswith("normalized_sharpness")]
        sharpness_scores["mean_sharpness"] = np.mean(all_sharpness) if all_sharpness else 0.0
        
        return sharpness_scores


class PredictionIntervalCoverage:
    """
    Métrica SOTA para coverage de intervalos de predição
    
    Verifica se a % real de targets dentro dos intervalos bate com a confiança esperada
    """
    
    def __call__(
        self, 
        predictions: BayesianPredictionOutput, 
        targets: torch.Tensor
    ) -> Dict[str, float]:
        """
        Calcula coverage para diferentes níveis de confiança
        
        Args:
            predictions: Predições Bayesianas
            targets: Valores reais
            
        Returns:
            Dict com coverage scores (próximo de conf_level = bem calibrado)
        """
        
        coverage_scores = {}
        
        for conf_key, interval in predictions.confidence_intervals.items():
            # Extrair nível de confiança do nome (ex: "95%" -> 0.95)
            expected_coverage = float(conf_key.replace("%", "")) / 100.0
            
            lower = interval["lower"]
            upper = interval["upper"]
            
            # Calcular coverage real
            in_interval = (targets >= lower) & (targets <= upper)
            actual_coverage = in_interval.float().mean()
            
            # Erro de coverage (ideal = 0)
            coverage_error = torch.abs(actual_coverage - expected_coverage)
            
            coverage_scores[f"coverage_{conf_key}"] = actual_coverage.item()
            coverage_scores[f"coverage_error_{conf_key}"] = coverage_error.item()
        
        # Coverage médio e erro médio
        all_coverage = [v for k, v in coverage_scores.items() if k.startswith("coverage_") and not "error" in k]
        all_errors = [v for k, v in coverage_scores.items() if "coverage_error" in k]
        
        coverage_scores["mean_coverage"] = np.mean(all_coverage) if all_coverage else 0.0
        coverage_scores["mean_coverage_error"] = np.mean(all_errors) if all_errors else 0.0
        
        return coverage_scores


class BayesianMetricsCalculator:
    """
    Calculadora unificada para todas as métricas Bayesianas SOTA
    """
    
    def __init__(self):
        self.calibration = BayesianCalibration()
        self.uncertainty_decomp = UncertaintyDecomposition()
        self.sharpness = PredictionSharpness()
        self.coverage = PredictionIntervalCoverage()
    
    def calculate_all_metrics(
        self, 
        predictions: BayesianPredictionOutput, 
        targets: torch.Tensor
    ) -> BayesianMetrics:
        """
        Calcula todas as métricas Bayesianas SOTA
        
        Args:
            predictions: Saída do modelo Bayesiano
            targets: Valores reais
            
        Returns:
            BayesianMetrics com todas as métricas calculadas
        """
        
        # Calcular métricas individuais
        ece = self.calibration(predictions, targets)
        coverage_dict = self.coverage(predictions, targets)
        uncertainty_dict = self.uncertainty_decomp(predictions)
        sharpness_dict = self.sharpness(predictions)
        
        # Calcular PIT (Probability Integral Transform)
        pit_histogram = self._calculate_pit(predictions, targets)
        
        # Compilar métricas finais
        return BayesianMetrics(
            ece=ece,
            coverage={k: v for k, v in coverage_dict.items() if k.startswith("coverage_") and not "error" in k},
            pit_histogram=pit_histogram,
            sharpness=sharpness_dict.get("mean_sharpness", 0.0),
            epistemic_ratio=uncertainty_dict.get("epistemic_ratio", 0.0),
            reliability_score=1.0 - ece  # Reliability = 1 - ECE
        )
    
    def _calculate_pit(
        self, 
        predictions: BayesianPredictionOutput, 
        targets: torch.Tensor,
        n_bins: int = 20
    ) -> np.ndarray:
        """
        Calcula Probability Integral Transform (PIT)
        
        Para predições bem calibradas, o histograma PIT deve ser uniforme
        """
        
        # Usar distribuição Student-T se disponível
        if predictions.student_t_params is not None:
            from torch.distributions import StudentT
            
            dist = StudentT(
                df=predictions.student_t_params["df"],
                loc=predictions.student_t_params["loc"],
                scale=predictions.student_t_params["scale"]
            )
            
            # Calcular CDFs (probabilidades acumuladas)
            pit_values = dist.cdf(targets)
        else:
            # Fallback: usar amostras MC para aproximar CDF
            mc_samples = predictions.mc_samples  # [batch, pred_len, n_samples]
            
            # Para cada target, calcular que % das amostras MC são menores
            pit_values = (mc_samples < targets.unsqueeze(-1)).float().mean(dim=-1)
        
        # Criar histograma
        pit_histogram, _ = np.histogram(pit_values.cpu().numpy().flatten(), bins=n_bins, range=(0, 1))
        
        return pit_histogram / np.sum(pit_histogram)  # Normalizar
    
    def create_diagnostic_plots(
        self, 
        metrics: BayesianMetrics, 
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Cria plots diagnósticos para análise visual da calibração
        """
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: PIT Histogram
        bins = np.arange(len(metrics.pit_histogram))
        axes[0, 0].bar(bins, metrics.pit_histogram, alpha=0.7)
        axes[0, 0].axhline(y=1/len(metrics.pit_histogram), color='red', linestyle='--', 
                          label='Uniform (well-calibrated)')
        axes[0, 0].set_title('PIT Histogram')
        axes[0, 0].set_xlabel('PIT Bin')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].legend()
        
        # Plot 2: Coverage por nível de confiança
        conf_levels = list(metrics.coverage.keys())
        coverage_values = [metrics.coverage[k] for k in conf_levels]
        expected_values = [float(k.replace("coverage_", "").replace("%", ""))/100 for k in conf_levels]
        
        axes[0, 1].scatter(expected_values, coverage_values, s=50, alpha=0.7)
        axes[0, 1].plot([0, 1], [0, 1], 'r--', label='Perfect Calibration')
        axes[0, 1].set_xlabel('Expected Coverage')
        axes[0, 1].set_ylabel('Actual Coverage')
        axes[0, 1].set_title('Coverage Calibration')
        axes[0, 1].legend()
        
        # Plot 3: Decomposição de incerteza
        uncertainty_types = ['Epistemic', 'Aleatoric']
        uncertainty_values = [metrics.epistemic_ratio, 1.0 - metrics.epistemic_ratio]
        colors = ['red', 'blue']
        
        axes[1, 0].pie(uncertainty_values, labels=uncertainty_types, colors=colors, autopct='%1.1f%%')
        axes[1, 0].set_title('Uncertainty Decomposition')
        
        # Plot 4: Métricas resumo
        metric_names = ['ECE', 'Reliability', 'Sharpness']
        metric_values = [metrics.ece, metrics.reliability_score, metrics.sharpness]
        
        bars = axes[1, 1].bar(metric_names, metric_values, alpha=0.7)
        axes[1, 1].set_title('Summary Metrics')
        axes[1, 1].set_ylabel('Score')
        
        # Colorir barras baseado na qualidade
        for bar, value in zip(bars, metric_values):
            if value > 0.8:  # Bom
                bar.set_color('green')
            elif value > 0.6:  # Médio
                bar.set_color('orange') 
            else:  # Ruim
                bar.set_color('red')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig


# Factory function para usar nas configurações
def create_bayesian_metrics():
    """Factory function para criar métricas Bayesianas"""
    calculator = BayesianMetricsCalculator()
    
    return {
        "calibration": calculator.calibration,
        "uncertainty_decomposition": calculator.uncertainty_decomp,
        "sharpness": calculator.sharpness,
        "coverage": calculator.coverage,
        "full_calculator": calculator
    }
