#!/usr/bin/env python3
"""
Teste SOTA do pipeline de fine-tuning
Valida se todos os componentes estão funcionando corretamente
"""

import sys
from pathlib import Path
import torch
import logging

# Setup paths
root_path = Path(__file__).parent
sys.path.append(str(root_path / "src"))

# Configure simple logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def test_imports():
    """Testa se todos os imports necessários funcionam"""
    print("🔍 Testando imports...")
    
    try:
        from uni2ts.model.moirai_moe import MoiraiMoEModule, MoiraiMoEForecast
        print("✅ MoiraiMoE imports OK")
    except Exception as e:
        print(f"❌ Erro MoiraiMoE: {e}")
        return False
    
    try:
        from uni2ts.model.crypto.bayesian_head import BayesianPredictionHead, BayesianPredictionOutput
        print("✅ BayesianPredictionHead imports OK")
    except Exception as e:
        print(f"❌ Erro BayesianHead: {e}")
        return False
    
    try:
        from uni2ts.loss.bayesian_elbo import BayesianELBOLoss
        print("✅ BayesianELBOLoss imports OK")
    except Exception as e:
        print(f"❌ Erro BayesianELBO: {e}")
        return False
    
    try:
        from uni2ts.data.builder.crypto import CryptoDatasetBuilder
        print("✅ CryptoDatasetBuilder imports OK")
    except Exception as e:
        print(f"❌ Erro CryptoDatasetBuilder: {e}")
        return False
    
    try:
        from uni2ts.callbacks.bayesian_uncertainty import BayesianUncertaintyMonitor
        print("✅ BayesianUncertaintyMonitor imports OK")
    except Exception as e:
        print(f"❌ Erro BayesianUncertaintyMonitor: {e}")
        return False
        
    return True

def test_config_loading():
    """Testa se a configuração YAML carrega corretamente"""
    print("\n🔍 Testando configuração YAML...")
    
    try:
        from omegaconf import OmegaConf
        config_path = Path("configs/crypto/finetune_bayesian_moe.yaml")
        
        if not config_path.exists():
            print(f"❌ Arquivo de configuração não encontrado: {config_path}")
            return False
        
        config = OmegaConf.load(config_path)
        
        # Validar seções essenciais
        required_sections = ['model', 'data', 'trainer', 'loss_func', 'callbacks']
        for section in required_sections:
            if section not in config:
                print(f"❌ Seção '{section}' não encontrada na configuração")
                return False
        
        # Validar callbacks especificam val/ELBO
        for callback in config.callbacks:
            if hasattr(callback, 'monitor') and 'val/ELBO' not in str(callback.monitor):
                print(f"⚠️ Callback monitora métrica incorreta: {callback.monitor}")
        
        print("✅ Configuração YAML OK")
        return True
        
    except Exception as e:
        print(f"❌ Erro ao carregar configuração: {e}")
        return False

def test_bayesian_components():
    """Testa se os componentes Bayesianos funcionam"""
    print("\n🔍 Testando componentes Bayesianos...")
    
    try:
        from uni2ts.model.crypto.bayesian_head import BayesianPredictionHead
        from uni2ts.loss.bayesian_elbo import BayesianELBOLoss
        
        # Testar BayesianPredictionHead
        head = BayesianPredictionHead(
            d_model=512,
            prediction_length=60,
            mc_dropout_rate=0.15,
            num_mc_samples=10,  # Reduzido para teste
            use_variational_weights=True,
            use_temporal_attention=True,
            student_t_df=4.0,
            confidence_levels=[0.8, 0.9, 0.95, 0.99]
        )
        
        # Teste forward
        batch_size = 2
        seq_len = 64  # Reduzido para teste
        d_model = 512
        
        representations = torch.randn(batch_size, seq_len, d_model)
        output = head(representations, training=True)
        
        # Verificar atributos obrigatórios
        required_attrs = ['epistemic_uncertainty', 'aleatoric_uncertainty', 'total_uncertainty']
        for attr in required_attrs:
            if not hasattr(output, attr):
                print(f"❌ Atributo '{attr}' não encontrado na saída Bayesiana")
                return False
        
        print(f"✅ BayesianPredictionHead OK - Shape: {output.mean_prediction.shape}")
        
        # Testar BayesianELBOLoss
        loss_fn = BayesianELBOLoss(
            likelihood_weight=1.0,
            kl_weight=1e-4,
            kl_annealing=True,
            kl_annealing_epochs=10,
            use_student_t=True,
            student_t_df_min=2.1
        )
        
        target = torch.randn(batch_size, 60)
        kl_divergence = head.get_kl_divergence()
        loss_dict = loss_fn(output, target, kl_divergence)
        
        # Verificar se retorna dicionário com loss
        if 'loss' not in loss_dict:
            print("❌ Loss function não retorna 'loss'")
            return False
        
        print(f"✅ BayesianELBOLoss OK - Loss: {loss_dict['loss']:.4f}")
        return True
        
    except Exception as e:
        print(f"❌ Erro nos componentes Bayesianos: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_callback():
    """Testa se o callback de incerteza funciona"""
    print("\n🔍 Testando callback de incerteza...")
    
    try:
        from uni2ts.callbacks.bayesian_uncertainty import BayesianUncertaintyMonitor
        from uni2ts.model.crypto.bayesian_head import BayesianPredictionOutput
        
        callback = BayesianUncertaintyMonitor(
            log_attention_weights=False,  # Simplificar teste
            uncertainty_threshold=0.05,
            log_interval=10,
            save_plots=False  # Não criar plots no teste
        )
        
        # Simular saída Bayesiana
        batch_size = 2
        prediction_length = 60
        
        mock_output = BayesianPredictionOutput(
            mean_prediction=torch.randn(batch_size, prediction_length),
            epistemic_uncertainty=torch.abs(torch.randn(batch_size, prediction_length)),
            aleatoric_uncertainty=torch.abs(torch.randn(batch_size, prediction_length)),
            total_uncertainty=torch.abs(torch.randn(batch_size, prediction_length)),
            confidence_intervals={},
            mc_samples=torch.randn(batch_size, prediction_length, 10),
            attention_weights=None,
            student_t_params=None
        )
        
        # Testar processamento
        callback._process_bayesian_output(mock_output, global_step=1)
        
        # Verificar se histórico foi atualizado
        if len(callback.uncertainty_history["epistemic"]) == 0:
            print("❌ Callback não registrou métricas")
            return False
        
        print("✅ BayesianUncertaintyMonitor OK")
        return True
        
    except Exception as e:
        print(f"❌ Erro no callback: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Executa todos os testes"""
    print("🚀 TESTE SOTA PIPELINE CRYPTO")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_config_loading,
        test_bayesian_components,
        test_callback
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n" + "=" * 50)
    print("📊 RESUMO DOS TESTES:")
    
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"✅ TODOS OS TESTES PASSARAM ({passed}/{total})")
        print("🎉 Pipeline SOTA está funcional!")
        return True
    else:
        print(f"❌ ALGUNS TESTES FALHARAM ({passed}/{total})")
        print("🔧 Pipeline precisa de correções")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
