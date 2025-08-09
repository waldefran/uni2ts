#!/usr/bin/env python3
"""
Script de Preparação de Dataset SOTA para Fine-tuning Crypto
Prepara dados do binanceDataloader.py para treinamento com todas as melhorias BOOST.MD

Uso:
    python scripts/crypto/prepare_dataset.py --data_path ./binance_data --output_path ./crypto_dataset
"""

import argparse
import sys
from pathlib import Path
import logging

# Adicionar src ao path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from uni2ts.data.builder.crypto import CryptoDatasetBuilder, CryptoConfig


def setup_logging():
    """Configura logging para o script"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('prepare_dataset.log')
        ]
    )
    return logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Preparar dataset crypto SOTA")
    parser.add_argument(
        "--data_path", 
        type=str, 
        default="./binance_data",
        help="Caminho para dados brutos do binanceDataloader.py"
    )
    parser.add_argument(
        "--output_path", 
        type=str, 
        default="./crypto_dataset",
        help="Caminho para salvar dataset processado"
    )
    parser.add_argument(
        "--assets", 
        nargs="+", 
        default=["BTCUSDT", "ETHUSDT", "ETHBTC", "BNBUSDT"],
        help="Lista de ativos para processar"
    )
    parser.add_argument(
        "--context_length", 
        type=int, 
        default=2048,
        help="Comprimento do contexto (minutos)"
    )
    parser.add_argument(
        "--prediction_length", 
        type=int, 
        default=60,
        help="Comprimento da predição (minutos)"
    )
    parser.add_argument(
        "--validation_split", 
        type=float, 
        default=0.2,
        help="Fração para validação (0.0-1.0)"
    )
    parser.add_argument(
        "--no_cyclical", 
        action="store_true",
        help="Desabilitar features cíclicas"
    )
    parser.add_argument(
        "--no_window_norm", 
        action="store_true",
        help="Desabilitar normalização por janela"
    )
    parser.add_argument(
        "--no_anonymous", 
        action="store_true",
        help="Manter item_id (não anônimo)"
    )
    
    args = parser.parse_args()
    logger = setup_logging()
    
    logger.info("🚀 Iniciando preparação de dataset crypto SOTA")
    logger.info(f"📂 Dados de entrada: {args.data_path}")
    logger.info(f"📂 Dados de saída: {args.output_path}")
    logger.info(f"📊 Ativos: {args.assets}")
    
    # Verificar se dados existem
    data_path = Path(args.data_path)
    if not data_path.exists():
        logger.error(f"❌ Caminho de dados não encontrado: {data_path}")
        return 1
    
    # Criar diretório de saída
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Configuração SOTA
    config = CryptoConfig(
        context_length=args.context_length,
        prediction_length=args.prediction_length,
        unified_dataset=True,                      # UPGRADE BOOST.MD
        anonymous_training=not args.no_anonymous,  # UPGRADE BOOST.MD
        window_normalization=not args.no_window_norm,  # UPGRADE BOOST.MD
        cyclical_features=not args.no_cyclical,   # UPGRADE BOOST.MD
        validation_split=args.validation_split,
        target_assets=args.assets
    )
    
    logger.info("⚙️ Configuração SOTA aplicada:")
    logger.info(f"   📊 Dataset unificado: {config.unified_dataset}")
    logger.info(f"   🎭 Treinamento anônimo: {config.anonymous_training}")
    logger.info(f"   📏 Normalização por janela: {config.window_normalization}")
    logger.info(f"   🔄 Features cíclicas: {config.cyclical_features}")
    logger.info(f"   📐 Contexto: {config.context_length} min")
    logger.info(f"   🎯 Predição: {config.prediction_length} min")
    
    try:
        # Criar builder
        builder = CryptoDatasetBuilder(data_path, config)
        
        # Info das features
        feature_info = builder.get_feature_info()
        logger.info("🔧 Features construídas:")
        for key, value in feature_info.items():
            logger.info(f"   {key}: {value}")
        
        # Construir datasets
        logger.info("🏗️ Construindo dataset de treinamento...")
        train_dataset = builder.build_dataset("train")
        
        logger.info("🏗️ Construindo dataset de validação...")
        val_dataset = builder.build_dataset("validation")
        
        # Salvar datasets
        train_path = output_path / "train"
        val_path = output_path / "validation"
        
        logger.info("💾 Salvando datasets...")
        train_dataset.save_to_disk(str(train_path))
        val_dataset.save_to_disk(str(val_path))
        
        # Salvar configuração
        config_path = output_path / "config.json"
        import json
        with open(config_path, 'w') as f:
            config_dict = {
                'context_length': config.context_length,
                'prediction_length': config.prediction_length,
                'unified_dataset': config.unified_dataset,
                'anonymous_training': config.anonymous_training,
                'window_normalization': config.window_normalization,
                'cyclical_features': config.cyclical_features,
                'target_assets': config.target_assets,
                'feature_info': feature_info
            }
            json.dump(config_dict, f, indent=2, default=str)
        
        # Estatísticas finais
        logger.info("✅ Dataset preparado com sucesso!")
        logger.info(f"   📈 Treinamento: {len(train_dataset):,} sequências")
        logger.info(f"   📊 Validação: {len(val_dataset):,} sequências")
        logger.info(f"   💾 Salvo em: {output_path}")
        logger.info(f"   ⚙️ Config: {config_path}")
        
        # Exemplo de uma sequência
        if len(train_dataset) > 0:
            sample = train_dataset[0]
            logger.info("📋 Exemplo de sequência:")
            logger.info(f"   Target shape: {len(sample['target'])} x {len(sample['target'][0])}")
            logger.info(f"   Future target shape: {len(sample['future_target'])}")
            logger.info(f"   Frequência: {sample['freq']}")
            logger.info(f"   Início: {sample['start']}")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Erro durante preparação: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    exit(main())
