#!/usr/bin/env python3
"""
RELATÓRIO FINAL: Estado do Ambiente SOTA
"""

import sys
from pathlib import Path

# Adicionar src ao path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def generate_environment_report():
    """Gera relatório completo do ambiente"""
    
    print("🏆 RELATÓRIO FINAL: AMBIENTE SOTA CRYPTO - UNI2TS")
    print("=" * 60)
    
    # Informações básicas
    import torch
    import pytorch_lightning as pl
    import numpy as np
    
    print(f"\n📋 INFORMAÇÕES DO AMBIENTE:")
    print(f"  🐍 Python: {sys.version.split()[0]}")
    print(f"  🔥 PyTorch: {torch.__version__}")
    print(f"  ⚡ Lightning: {pl.__version__}")
    print(f"  🔢 NumPy: {np.__version__}")
    print(f"  🖥️  CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  🎮 GPU: {torch.cuda.get_device_name(0)}")
        print(f"  💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    print(f"\n🏗️ COMPONENTES SOTA IMPLEMENTADOS:")
    
    # Testar cada componente
    components = [
        ("BayesianPredictionHead", "uni2ts.model.crypto.bayesian_head", "BayesianPredictionHead"),
        ("BayesianLinear", "uni2ts.model.crypto.bayesian_head", "BayesianLinear"),
        ("BayesianELBOLoss", "uni2ts.loss.bayesian_elbo", "BayesianELBOLoss"),
        ("CryptoDatasetBuilder", "uni2ts.data.builder.crypto", "CryptoDatasetBuilder"),
        ("BayesianUncertaintyMonitor", "uni2ts.callbacks.bayesian_uncertainty", "BayesianUncertaintyMonitor"),
        ("BayesianMetricsCalculator", "uni2ts.eval_util.bayesian_metrics", "BayesianMetricsCalculator"),
    ]
    
    working_components = []
    failed_components = []
    
    for name, module, class_name in components:
        try:
            exec(f"from {module} import {class_name}")
            working_components.append(name)
            print(f"  ✅ {name}")
        except Exception as e:
            failed_components.append((name, str(e)))
            print(f"  ❌ {name}: {e}")
    
    print(f"\n📊 ARQUIVOS DE CONFIGURAÇÃO:")
    
    config_files = [
        ("Config SOTA", "configs/crypto/finetune_bayesian_moe.yaml"),
        ("Crypto README", "CRYPTO_README.md"),
        ("Kaggle Tutorial", "KAGGLE_FINETUNING_TUTORIAL.md"),
        ("Implementation Status", "IMPLEMENTATION_STATUS.md"),
        ("Requirements", "requirements_crypto.txt"),
        ("Environment", ".env"),
    ]
    
    for name, filepath in config_files:
        if Path(filepath).exists():
            print(f"  ✅ {name}: {filepath}")
        else:
            print(f"  ❌ {name}: {filepath} (not found)")
    
    print(f"\n🧪 SCRIPTS DE TESTE E VALIDAÇÃO:")
    
    test_scripts = [
        ("Teste Simplificado", "test_simple.py"),
        ("Teste Completo", "test_environment.py"),
        ("Validação SOTA", "scripts/crypto/validate_sota_pipeline.py"),
        ("Preparação Dataset", "scripts/crypto/prepare_dataset.py"),
        ("Fine-tuning Script", "scripts/crypto/finetune_model.py"),
    ]
    
    for name, filepath in test_scripts:
        if Path(filepath).exists():
            print(f"  ✅ {name}: {filepath}")
        else:
            print(f"  ❌ {name}: {filepath} (not found)")
    
    print(f"\n🚀 CLI E INTERFACES:")
    try:
        from uni2ts.cli.crypto_bayesian import main as crypto_cli
        print(f"  ✅ CLI Crypto Bayesiano: uni2ts.cli.crypto_bayesian")
    except Exception as e:
        print(f"  ❌ CLI Crypto Bayesiano: {e}")
    
    print(f"\n📈 RESUMO FINAL:")
    print(f"  ✅ Componentes funcionais: {len(working_components)}/{len(components)}")
    print(f"  📁 Arquivos de config: {sum(1 for _, path in config_files if Path(path).exists())}/{len(config_files)}")
    print(f"  🧪 Scripts disponíveis: {sum(1 for _, path in test_scripts if Path(path).exists())}/{len(test_scripts)}")
    
    if len(working_components) == len(components):
        print(f"\n🎉 AMBIENTE SOTA COMPLETAMENTE FUNCIONAL!")
        print(f"📋 PRÓXIMOS PASSOS:")
        print(f"  1. python test_simple.py  # Teste rápido")
        print(f"  2. python scripts/crypto/validate_sota_pipeline.py  # Validação completa")
        print(f"  3. python -m uni2ts.cli.crypto_bayesian --help  # CLI avançado")
        print(f"  4. Seguir KAGGLE_FINETUNING_TUTORIAL.md para fine-tuning")
        
        return True
    else:
        print(f"\n⚠️ ALGUNS COMPONENTES PRECISAM DE ATENÇÃO:")
        for name, error in failed_components:
            print(f"  ❌ {name}: {error}")
        
        return False

if __name__ == "__main__":
    success = generate_environment_report()
    sys.exit(0 if success else 1)
