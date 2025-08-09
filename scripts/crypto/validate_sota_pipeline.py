#!/usr/bin/env python3
"""
Script de Validação SOTA para Fine-tuning Crypto Bayesiano
Testa todos os componentes implementados antes do treinamento real

Uso:
    python scripts/crypto/validate_sota_pipeline.py
"""

import sys
import os
from pathlib import Path
import logging
import traceback

# Adicionar src ao path
root_path = Path(__file__).parent.parent.parent
sys.path.append(str(root_path))
sys.path.append(str(root_path / "src"))

def setup_logging():
    """Configura logging para validação"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def test_imports():
    """Testa imports de todos os componentes SOTA"""
    print("🔍 Testando imports dos componentes SOTA...")
    
    try:
        # Core Bayesiano
        from uni2ts.model.crypto.bayesian_head import BayesianPredictionHead, BayesianLinear
        from uni2ts.loss.bayesian_elbo import BayesianELBOLoss
        print("  ✅ Componentes Bayesianos importados")
        
        # Dataset Builder
        from uni2ts.data.builder.crypto import CryptoDatasetBuilder, CryptoConfig
        print("  ✅ Dataset Builder importado")
        
        # Callbacks e Métricas
        from uni2ts.callbacks.bayesian_uncertainty import BayesianUncertaintyMonitor
        print("  ✅ Callbacks importados")
        
        # CLI
        from uni2ts.cli.crypto_bayesian import validate_crypto_bayesian_config
        print("  ✅ CLI integração importada")
        
        return True
        
    except ImportError as e:
        print(f"  ❌ Erro de import: {e}")
        return False

def test_bayesian_head():
    """Testa inicialização da cabeça Bayesiana"""
    print("🧠 Testando BayesianPredictionHead...")
    
    try:
        import torch
        from uni2ts.model.crypto.bayesian_head import BayesianPredictionHead
        
        # Configurações de teste
        d_model = 512
        prediction_length = 60
        
        # Inicializar cabeça
        head = BayesianPredictionHead(
            d_model=d_model,
            prediction_length=prediction_length,
            mc_dropout_rate=0.15,
            num_mc_samples=10,  # Reduzido para teste
            use_variational_weights=True,
            use_temporal_attention=True
        )
        
        # Teste forward pass
        batch_size = 2
        seq_len = 128
        reprs = torch.randn(batch_size, seq_len, d_model)
        
        output = head(reprs, training=False)
        
        # Validar saída
        assert hasattr(output, 'mean_prediction')
        assert hasattr(output, 'epistemic_uncertainty')
        assert hasattr(output, 'aleatoric_uncertainty')
        assert hasattr(output, 'total_uncertainty')
        assert hasattr(output, 'confidence_intervals')
        
        print(f"  ✅ Forward pass OK - Shape: {output.mean_prediction.shape}")
        print(f"  ✅ Incerteza epistêmica: {output.epistemic_uncertainty.mean():.4f}")
        print(f"  ✅ Incerteza aleatória: {output.aleatoric_uncertainty.mean():.4f}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Erro no BayesianPredictionHead: {e}")
        traceback.print_exc()
        return False

def test_elbo_loss():
    """Testa ELBO Loss"""
    print("📉 Testando BayesianELBOLoss...")
    
    try:
        import torch
        from uni2ts.loss.bayesian_elbo import BayesianELBOLoss
        from uni2ts.model.crypto.bayesian_head import BayesianPredictionOutput
        
        # Criar loss
        loss_fn = BayesianELBOLoss(
            likelihood_weight=1.0,
            kl_weight=1e-4,
            use_student_t=True
        )
        
        # Mock de predição Bayesiana
        batch_size, pred_len = 2, 60
        
        mock_output = BayesianPredictionOutput(
            mean_prediction=torch.randn(batch_size, pred_len),
            epistemic_uncertainty=torch.abs(torch.randn(batch_size, pred_len)) * 0.1,
            aleatoric_uncertainty=torch.abs(torch.randn(batch_size, pred_len)) * 0.1,
            total_uncertainty=torch.abs(torch.randn(batch_size, pred_len)) * 0.2,
            confidence_intervals={},
            mc_samples=torch.randn(batch_size, pred_len, 10),
            student_t_params={
                "loc": torch.randn(batch_size, pred_len),
                "scale": torch.abs(torch.randn(batch_size, pred_len)) + 0.1,
                "df": torch.ones(batch_size, pred_len) * 4.0
            }
        )
        
        targets = torch.randn(batch_size, pred_len)
        
        # Mock de modelo com KL divergence
        class MockModel:
            def get_kl_divergence(self):
                return torch.tensor(0.01)
        
        model = MockModel()
        
        # Calcular loss
        loss = loss_fn(mock_output, targets, model)
        
        print(f"  ✅ ELBO Loss calculado: {loss.item():.4f}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Erro no BayesianELBOLoss: {e}")
        traceback.print_exc()
        return False

def test_dataset_config():
    """Testa configuração do dataset"""
    print("📊 Testando CryptoConfig...")
    
    try:
        from uni2ts.data.builder.crypto import CryptoConfig
        
        # Criar configuração
        config = CryptoConfig(
            context_length=2048,
            prediction_length=60,
            unified_dataset=True,
            anonymous_training=True,
            window_normalization=True,
            cyclical_features=True
        )
        
        print(f"  ✅ Config criado - Assets: {config.target_assets}")
        print(f"  ✅ Dataset unificado: {config.unified_dataset}")
        print(f"  ✅ Treinamento anônimo: {config.anonymous_training}")
        print(f"  ✅ Normalização por janela: {config.window_normalization}")
        print(f"  ✅ Features cíclicas: {config.cyclical_features}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Erro no CryptoConfig: {e}")
        traceback.print_exc()
        return False

def test_callbacks():
    """Testa callbacks de monitoramento"""
    print("📈 Testando BayesianUncertaintyMonitor...")
    
    try:
        from uni2ts.callbacks.bayesian_uncertainty import BayesianUncertaintyMonitor
        
        # Criar callback
        monitor = BayesianUncertaintyMonitor(
            log_attention_weights=True,
            uncertainty_threshold=0.05,
            log_interval=10,
            save_plots=False  # Desabilitar para teste
        )
        
        print(f"  ✅ Callback criado - Threshold: {monitor.uncertainty_threshold}")
        print(f"  ✅ Log attention: {monitor.log_attention_weights}")
        print(f"  ✅ Histórico inicializado: {len(monitor.uncertainty_history['total'])}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Erro no BayesianUncertaintyMonitor: {e}")
        traceback.print_exc()
        return False

def test_config_loading():
    """Testa carregamento da configuração YAML"""
    print("⚙️  Testando carregamento da configuração...")
    
    try:
        from omegaconf import OmegaConf
        
        config_path = root_path / "configs" / "crypto" / "finetune_bayesian_moe.yaml"
        
        if not config_path.exists():
            print(f"  ❌ Config não encontrado: {config_path}")
            return False
        
        # Carregar config
        cfg = OmegaConf.load(config_path)
        
        # Validar componentes essenciais
        assert "model" in cfg
        assert "data" in cfg
        assert "trainer" in cfg
        assert "callbacks" in cfg
        assert "metrics" in cfg
        
        # Validar configurações Bayesianas
        if cfg.model.get("prediction_head"):
            head_target = cfg.model.prediction_head._target_
            assert "BayesianPredictionHead" in head_target
            print(f"  ✅ Cabeça Bayesiana configurada: {head_target}")
        
        if cfg.get("loss_func"):
            loss_target = cfg.loss_func._target_
            assert "BayesianELBOLoss" in loss_target
            print(f"  ✅ Loss Bayesiana configurada: {loss_target}")
        
        # Validar callbacks
        callback_targets = [cb._target_ for cb in cfg.callbacks]
        has_uncertainty_monitor = any("BayesianUncertaintyMonitor" in target for target in callback_targets)
        
        if has_uncertainty_monitor:
            print("  ✅ Callback de incerteza configurado")
        else:
            print("  ⚠️  Callback de incerteza não encontrado")
        
        print(f"  ✅ Configuração carregada - {len(cfg.callbacks)} callbacks, {len(cfg.get('metrics', []))} métricas")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Erro no carregamento da config: {e}")
        traceback.print_exc()
        return False

def test_dependencies():
    """Testa dependências essenciais"""
    print("📦 Testando dependências...")
    
    required_packages = [
        'torch',
        'pytorch_lightning', 
        'numpy',
        'pandas',
        'omegaconf'
    ]
    
    all_ok = True
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package} - FALTANDO")
            all_ok = False
    
    return all_ok

def main():
    """Executa todos os testes de validação"""
    setup_logging()
    
    print("🚀 VALIDAÇÃO DO PIPELINE SOTA CRYPTO BAYESIANO")
    print("=" * 60)
    
    tests = [
        ("Dependências", test_dependencies),
        ("Imports", test_imports),
        ("Configuração YAML", test_config_loading),
        ("Dataset Config", test_dataset_config),
        ("Bayesian Head", test_bayesian_head),
        ("ELBO Loss", test_elbo_loss),
        ("Callbacks", test_callbacks),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"  ❌ ERRO CRÍTICO: {e}")
            results[test_name] = False
    
    # Resumo final
    print("\n" + "="*60)
    print("📊 RESUMO DA VALIDAÇÃO:")
    print("="*60)
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    
    for test_name, passed in results.items():
        status = "✅ PASSOU" if passed else "❌ FALHOU"
        print(f"  {test_name:<20} {status}")
    
    print(f"\n📈 RESULTADO: {passed_tests}/{total_tests} testes passaram")
    
    if passed_tests == total_tests:
        print("🎉 TODOS OS COMPONENTES SOTA ESTÃO FUNCIONAIS!")
        print("✅ Pipeline pronto para fine-tuning crypto Bayesiano")
        return 0
    else:
        print("⚠️  ALGUNS COMPONENTES PRECISAM DE ATENÇÃO")
        print("❌ Corrija os erros antes de prosseguir")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
