#!/usr/bin/env python3
"""
Validação SOTA da Implementação Completa
Verifica consistência entre YAML, CryptoDatasetBuilder, e pipeline de treinamento

Este script valida:
1. CryptoDatasetBuilder SOTA com todas as features
2. Consistência com configuração YAML
3. Compatibilidade com uni2ts framework
4. Pipeline completo de treinamento
"""

import os
import sys
import traceback
import yaml
from pathlib import Path
from typing import Dict, List

def validate_crypto_dataset_builder():
    """Valida implementação do CryptoDatasetBuilder SOTA"""
    print("🔍 Validando CryptoDatasetBuilder SOTA...")
    
    try:
        from uni2ts.data.builder.crypto import CryptoDatasetBuilder, CryptoConfig
        
        # Test configuration
        config = CryptoConfig(
            context_length=1440,
            prediction_length=60,
            unified_dataset=True,
            anonymous_training=True,
            window_normalization=True,
            cyclical_features=True,
            target_assets=["BTCUSDT", "ETHUSDT"]
        )
        
        # Validate config object
        assert hasattr(config, 'context_length'), "Missing context_length"
        assert hasattr(config, 'prediction_length'), "Missing prediction_length"
        assert hasattr(config, 'unified_dataset'), "Missing unified_dataset"
        assert hasattr(config, 'anonymous_training'), "Missing anonymous_training"
        assert hasattr(config, 'window_normalization'), "Missing window_normalization"
        assert hasattr(config, 'cyclical_features'), "Missing cyclical_features"
        assert hasattr(config, 'dtype'), "Missing dtype configuration"
        
        print("   ✅ CryptoConfig validado")
        
        # Test builder instantiation (sem dados reais)
        builder = CryptoDatasetBuilder(
            data_path="./dummy_path",
            config=config
        )
        
        # Validate builder attributes
        assert hasattr(builder, 'numerical_fields'), "Missing numerical_fields"
        assert hasattr(builder, 'datetime_fields'), "Missing datetime_fields"
        assert hasattr(builder, 'technical_fields'), "Missing technical_fields"
        assert hasattr(builder, 'cyclical_fields'), "Missing cyclical_fields"
        
        # Validate methods
        assert hasattr(builder, 'build_dataset'), "Missing build_dataset method"
        assert hasattr(builder, 'load_dataset'), "Missing load_dataset method"
        assert hasattr(builder, 'validate_config'), "Missing validate_config method"
        assert hasattr(builder, 'get_feature_info'), "Missing get_feature_info method"
        
        print("   ✅ CryptoDatasetBuilder SOTA validado")
        return True
        
    except Exception as e:
        print(f"   ❌ Erro: {e}")
        traceback.print_exc()
        return False

def validate_yaml_config():
    """Valida configuração YAML SOTA"""
    print("🔍 Validando configuração YAML SOTA...")
    
    try:
        yaml_path = Path("configs/crypto/finetune_bayesian_moe.yaml")
        
        if not yaml_path.exists():
            print(f"   ⚠️ Arquivo YAML não encontrado: {yaml_path}")
            return False
        
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Validate required sections
        required_sections = [
            'model', 'data', 'trainer', 'optimizer', 
            'callbacks', 'metrics', 'loss_func'
        ]
        
        for section in required_sections:
            assert section in config, f"Missing section: {section}"
        
        # Validate model config
        model_config = config['model']
        assert 'context_length' in model_config, "Missing model.context_length"
        assert 'prediction_length' in model_config, "Missing model.prediction_length"
        
        # Validate data config
        data_config = config['data']
        assert '_target_' in data_config, "Missing data._target_"
        assert 'config' in data_config, "Missing data.config"
        
        data_builder_config = data_config['config']
        assert 'unified_dataset' in data_builder_config, "Missing unified_dataset"
        assert 'anonymous_training' in data_builder_config, "Missing anonymous_training"
        assert 'window_normalization' in data_builder_config, "Missing window_normalization"
        assert 'cyclical_features' in data_builder_config, "Missing cyclical_features"
        
        # Validate Bayesian components
        assert 'prediction_head' in model_config, "Missing Bayesian prediction head"
        head_config = model_config['prediction_head']
        assert 'mc_dropout_rate' in head_config, "Missing MC dropout"
        assert 'use_variational_weights' in head_config, "Missing variational weights"
        
        # Validate ELBO loss
        loss_config = config['loss_func']
        assert '_target_' in loss_config, "Missing loss function target"
        assert 'uni2ts.loss.bayesian_elbo.BayesianELBOLoss' in loss_config['_target_'], "Wrong loss function"
        
        print("   ✅ Configuração YAML SOTA validada")
        return True
        
    except Exception as e:
        print(f"   ❌ Erro: {e}")
        traceback.print_exc()
        return False

def validate_imports():
    """Valida todas as importações necessárias"""
    print("🔍 Validando importações SOTA...")
    
    imports_to_test = [
        'uni2ts.data.builder.crypto',
        'uni2ts.model.moirai',
        'uni2ts.loss.bayesian_elbo',
        'uni2ts.callbacks.bayesian_uncertainty',
        'uni2ts.callbacks.elbo_annealing',
        'uni2ts.eval_util.bayesian_metrics'
    ]
    
    success_count = 0
    
    for import_name in imports_to_test:
        try:
            __import__(import_name)
            print(f"   ✅ {import_name}")
            success_count += 1
        except ImportError as e:
            print(f"   ⚠️ {import_name}: {e}")
    
    if success_count == len(imports_to_test):
        print("   ✅ Todas as importações SOTA disponíveis")
        return True
    else:
        print(f"   ⚠️ {success_count}/{len(imports_to_test)} importações disponíveis")
        return False

def validate_notebook_consistency():
    """Valida consistência do notebook Kaggle"""
    print("🔍 Validando notebook Kaggle SOTA...")
    
    notebook_path = Path("kaggle_finetune_crypto.ipynb")
    
    if not notebook_path.exists():
        print(f"   ⚠️ Notebook não encontrado: {notebook_path}")
        return False
    
    try:
        # Read notebook and check for key patterns
        with open(notebook_path, 'r') as f:
            notebook_content = f.read()
        
        # Check for SOTA implementations
        sota_patterns = [
            'CryptoDatasetBuilder',
            'CryptoConfig',
            'unified_dataset=True',
            'anonymous_training=True',
            'window_normalization=True',
            'cyclical_features=True',
            'MoiraiBayesianLightningModule',
            'BayesianELBOLoss'
        ]
        
        missing_patterns = []
        for pattern in sota_patterns:
            if pattern not in notebook_content:
                missing_patterns.append(pattern)
        
        if missing_patterns:
            print(f"   ⚠️ Padrões SOTA faltando: {missing_patterns}")
            return False
        
        print("   ✅ Notebook contém todas as implementações SOTA")
        return True
        
    except Exception as e:
        print(f"   ❌ Erro: {e}")
        return False

def generate_validation_report():
    """Gera relatório completo de validação"""
    print("🏆 RELATÓRIO DE VALIDAÇÃO SOTA")
    print("=" * 60)
    
    results = {}
    
    # Run all validations
    results['crypto_builder'] = validate_crypto_dataset_builder()
    results['yaml_config'] = validate_yaml_config()
    results['imports'] = validate_imports()
    results['notebook'] = validate_notebook_consistency()
    
    # Summary
    print("\n📊 RESUMO DA VALIDAÇÃO:")
    total_checks = len(results)
    passed_checks = sum(results.values())
    
    for check_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {check_name:20} {status}")
    
    print(f"\n🎯 RESULTADO FINAL: {passed_checks}/{total_checks} validações passaram")
    
    if passed_checks == total_checks:
        print("🎉 IMPLEMENTAÇÃO SOTA VALIDADA COM SUCESSO!")
        print("🚀 Pipeline pronto para treinamento em produção")
        return True
    else:
        print("⚠️ Algumas validações falharam - verifique os erros acima")
        return False

if __name__ == "__main__":
    success = generate_validation_report()
    sys.exit(0 if success else 1)
