#!/usr/bin/env python3
"""
Script de Teste do Ambiente SOTA
Verifica se todos os componentes implementados estão funcionais
"""

import sys
import traceback
from pathlib import Path

# Adicionar src ao path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_imports():
    """Testar todas as importações críticas"""
    print("🔍 Testando importações...")
    
    tests = [
        ("torch", "torch"),
        ("numpy", "numpy"),
        ("lightning", "lightning"),
        ("gluonts", "gluonts"),
        ("jaxtyping", "jaxtyping"),
        ("hydra", "hydra"),
        ("yaml", "yaml"),
    ]
    
    for name, module in tests:
        try:
            __import__(module)
            print(f"  ✅ {name}")
        except ImportError as e:
            print(f"  ❌ {name}: {e}")
            return False
    
    return True

def test_uni2ts_core():
    """Testar componentes core do uni2ts"""
    print("\n🧠 Testando componentes uni2ts core...")
    
    try:
        from uni2ts.common.torch_util import safe_div, size_to_mask
        print("  ✅ torch_util")
        
        from uni2ts.common.env import env
        print("  ✅ env")
        
        # Testar funções básicas
        import torch
        
        # Teste safe_div
        result = safe_div(torch.tensor(1.0), torch.tensor(0.0))
        assert result == 1.0
        print("  ✅ safe_div funcionando")
        
        # Teste size_to_mask
        mask = size_to_mask(5, torch.tensor([3, 2]))
        expected_shape = (2, 5)
        assert mask.shape == expected_shape
        print("  ✅ size_to_mask funcionando")
        
        return True
    except Exception as e:
        print(f"  ❌ Erro no uni2ts core: {e}")
        traceback.print_exc()
        return False

def test_sota_components():
    """Testar componentes SOTA implementados"""
    print("\n🚀 Testando componentes SOTA...")
    
    try:
        # Testar Bayesian Head
        from uni2ts.model.crypto.bayesian_head import BayesianPredictionHead, BayesianLinear
        print("  ✅ BayesianPredictionHead importado")
        
        # Testar ELBO Loss
        from uni2ts.loss.bayesian_elbo import BayesianELBOLoss
        print("  ✅ BayesianELBOLoss importado")
        
        # Testar Dataset Builder
        from uni2ts.data.builder.crypto import CryptoDatasetBuilder
        print("  ✅ CryptoDatasetBuilder importado")
        
        # Testar Callbacks
        from uni2ts.callbacks.bayesian_uncertainty import BayesianUncertaintyMonitor
        print("  ✅ BayesianUncertaintyMonitor importado")
        
        # Testar Métricas
        from uni2ts.eval_util.bayesian_metrics import BayesianMetricsCalculator
        print("  ✅ BayesianMetricsCalculator importado")
        
        # Testar CLI
        from uni2ts.cli.crypto_bayesian import main as crypto_cli_main
        print("  ✅ CLI crypto_bayesian importado")
        
        return True
    except Exception as e:
        print(f"  ❌ Erro nos componentes SOTA: {e}")
        traceback.print_exc()
        return False

def test_basic_functionality():
    """Testar funcionalidade básica dos componentes"""
    print("\n⚡ Testando funcionalidade básica...")
    
    try:
        import torch
        from uni2ts.model.crypto.bayesian_head import BayesianLinear
        
        # Testar BayesianLinear
        layer = BayesianLinear(10, 5, use_variational=True)
        x = torch.randn(2, 10)
        output, kl_div = layer(x)
        
        assert output.shape == (2, 5)
        assert isinstance(kl_div, torch.Tensor)
        print("  ✅ BayesianLinear funcionando")
        
        # Testar ELBO Loss
        from uni2ts.loss.bayesian_elbo import BayesianELBOLoss
        loss_fn = BayesianELBOLoss()
        
        # Dados simulados
        predictions = torch.randn(2, 5)
        targets = torch.randn(2, 5)
        kl_divergences = [torch.tensor(0.1), torch.tensor(0.2)]
        
        loss = loss_fn(predictions, targets, kl_divergences)
        assert isinstance(loss, torch.Tensor)
        print("  ✅ BayesianELBOLoss funcionando")
        
        # Testar Métricas
        from uni2ts.eval_util.bayesian_metrics import BayesianCalibration
        calibration = BayesianCalibration()
        print("  ✅ BayesianCalibration inicializado")
        
        return True
    except Exception as e:
        print(f"  ❌ Erro na funcionalidade básica: {e}")
        traceback.print_exc()
        return False

def test_config_loading():
    """Testar carregamento de configurações"""
    print("\n📋 Testando configurações...")
    
    try:
        import yaml
        config_path = Path("configs/crypto/finetune_bayesian_moe.yaml")
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            print("  ✅ Configuração SOTA carregada")
            
            # Verificar campos essenciais
            essential_fields = ['model', 'loss_func', 'data']
            for field in essential_fields:
                if field in config:
                    print(f"    ✅ {field}")
                else:
                    print(f"    ❌ {field} não encontrado")
                    return False
        else:
            print("  ⚠️ Arquivo de configuração não encontrado")
            return True  # Não é crítico para o teste
        
        return True
    except Exception as e:
        print(f"  ❌ Erro no carregamento de config: {e}")
        return False

def main():
    """Função principal de teste"""
    print("🧪 TESTE DO AMBIENTE SOTA - Uni2TS Crypto\n")
    
    tests = [
        ("Importações Básicas", test_imports),
        ("Uni2TS Core", test_uni2ts_core),
        ("Componentes SOTA", test_sota_components),
        ("Funcionalidade Básica", test_basic_functionality),
        ("Configurações", test_config_loading),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"🧪 {test_name}")
        print('='*50)
        
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ Erro inesperado em {test_name}: {e}")
            results.append((test_name, False))
    
    # Resumo final
    print(f"\n{'='*50}")
    print("📊 RESUMO DOS TESTES")
    print('='*50)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASSOU" if success else "❌ FALHOU"
        print(f"{test_name:.<30} {status}")
        if success:
            passed += 1
    
    print(f"\n🎯 Resultado: {passed}/{total} testes passaram")
    
    if passed == total:
        print("🎉 TODOS OS TESTES PASSARAM! Ambiente SOTA está funcional! 🚀")
        return 0
    else:
        print("⚠️ Alguns testes falharam. Verifique os detalhes acima.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
