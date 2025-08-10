#!/usr/bin/env python3
"""
Teste Simples dos Componentes SOTA
"""

import sys
import os
from pathlib import Path

# Adicionar src ao path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_basic_imports():
    """Teste básico de importações"""
    print("🔍 Testando importações básicas...")
    
    try:
        import torch
        print(f"  ✅ PyTorch {torch.__version__}")
        
        import numpy as np
        print(f"  ✅ NumPy {np.__version__}")
        
        import pytorch_lightning as pl
        print(f"  ✅ PyTorch Lightning {pl.__version__}")
        
        return True
    except Exception as e:
        print(f"  ❌ Erro nas importações básicas: {e}")
        return False

def test_bayesian_head():
    """Testar o BayesianPredictionHead"""
    print("\n🧠 Testando Bayesian Head...")
    
    try:
        from uni2ts.model.crypto.bayesian_head import BayesianLinear, BayesianPredictionHead
        print("  ✅ Importação bem sucedida")
        
        # Teste funcional
        import torch
        layer = BayesianLinear(10, 5)
        x = torch.randn(2, 10)
        output, kl = layer(x)
        kl_scalar = kl.mean().item() if kl.numel() > 1 else kl.item()
        print(f"  ✅ BayesianLinear funcional: output {output.shape}, kl {kl_scalar:.4f}")
        
        return True
    except Exception as e:
        print(f"  ❌ Erro: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_elbo_loss():
    """Testar BayesianELBOLoss"""
    print("\n💯 Testando ELBO Loss...")
    
    try:
        from uni2ts.loss.bayesian_elbo import BayesianELBOLoss
        print("  ✅ Importação bem sucedida")
        
        # Teste funcional
        import torch
        from uni2ts.model.crypto.bayesian_head import BayesianPredictionOutput
        
        loss_fn = BayesianELBOLoss()
        
        # Criar objeto BayesianPredictionOutput adequado
        mean_pred = torch.randn(2, 5)
        epistemic = torch.randn(2, 5).abs()
        aleatoric = torch.randn(2, 5).abs()
        total = epistemic + aleatoric
        
        prediction_output = BayesianPredictionOutput(
            mean_prediction=mean_pred,
            epistemic_uncertainty=epistemic,
            aleatoric_uncertainty=aleatoric,
            total_uncertainty=total,
            confidence_intervals={},
            mc_samples=torch.randn(2, 5, 100),
            student_t_params=None
        )
        
        targets = torch.randn(2, 5)
        kl_divs_list = [torch.tensor(0.1), torch.tensor(0.2)]
        kl_divergence = sum(kl_divs_list)  # Somar os KL divergences
        
        loss = loss_fn(prediction_output, targets, kl_divergence)
        print(f"  ✅ BayesianELBOLoss funcional: loss {loss['loss'].item():.4f}")
        
        return True
    except Exception as e:
        print(f"  ❌ Erro: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_metrics():
    """Testar métricas Bayesianas"""
    print("\n📊 Testando Métricas Bayesianas...")
    
    try:
        # Configurar matplotlib
        import matplotlib
        matplotlib.use('Agg')
        
        from uni2ts.eval_util.bayesian_metrics import BayesianCalibration, BayesianMetricsCalculator
        print("  ✅ Importação bem sucedida")
        
        # Teste funcional básico
        calibration = BayesianCalibration()
        calculator = BayesianMetricsCalculator()
        print("  ✅ Objetos criados com sucesso")
        
        return True
    except Exception as e:
        print(f"  ❌ Erro: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dataset_builder():
    """Testar CryptoDatasetBuilder"""
    print("\n🗂️ Testando Dataset Builder...")
    
    try:
        from uni2ts.data.builder.crypto import CryptoDatasetBuilder
        print("  ✅ Importação bem sucedida")
        
        return True
    except Exception as e:
        print(f"  ❌ Erro: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Função principal"""
    print("🧪 TESTE SIMPLIFICADO DOS COMPONENTES SOTA\n")
    
    tests = [
        ("Importações Básicas", test_basic_imports),
        ("Bayesian Head", test_bayesian_head),
        ("ELBO Loss", test_elbo_loss),
        ("Métricas Bayesianas", test_metrics),
        ("Dataset Builder", test_dataset_builder),
    ]
    
    results = []
    for name, test_func in tests:
        success = test_func()
        results.append((name, success))
    
    # Resumo
    print(f"\n{'='*50}")
    print("📊 RESUMO FINAL")
    print('='*50)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for name, success in results:
        status = "✅ PASSOU" if success else "❌ FALHOU"
        print(f"{name:.<30} {status}")
    
    print(f"\n🎯 Resultado: {passed}/{total} testes passaram")
    
    if passed == total:
        print("🎉 TODOS OS COMPONENTES SOTA ESTÃO FUNCIONAIS! 🚀")
        return 0
    else:
        print("⚠️ Alguns componentes precisam de ajustes.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
