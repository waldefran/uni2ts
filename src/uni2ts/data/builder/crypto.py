"""
Crypto Dataset Builder SOTA - Dataset Unificado e Anônimo
Implementa todas as melhorias do BOOST.MD para estado da arte

FILOSOFIA SOTA: "TUDO JÁ ESTÁ NO PREÇO"
- Usar apenas dados essenciais da Binance (OHLCV + metadados)
- Features temporais cíclicas para contexto
- Minimal feature engineering - deixar o modelo aprender padrões
- Foco em qualidade dos dados, não quantidade de features

UPGRADE BOOST.MD implementado:
- Dataset unificado: todos os ativos em um só dataset
- Anonimização: remove item_id durante treinamento
- Normalização por janela: (valor[t] / valor[t=0]) - 1
- Features cíclicas explícitas: sin/cos para minute, hour, weekday
- Features técnicas essenciais: apenas returns e ratios básicos
- Compatível com Moirai-MoE architecture
"""

import pandas as pd
import numpy as np
import torch
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union, Set
from datetime import datetime
import pickle
from datasets import Dataset, Features, Value, Sequence
from dataclasses import dataclass, field

from uni2ts.data.builder import DatasetBuilder


@dataclass
class CryptoConfig:
    """Configuração para processamento de dados crypto"""
    context_length: int = 2048      # 2048 minutos de contexto (~34 horas)
    prediction_length: int = 60     # Próximos 60 minutos
    unified_dataset: bool = True    # Dataset unificado (BOOST.MD)
    anonymous_training: bool = True # Remove item_id (BOOST.MD)
    window_normalization: bool = True # (valor[t] / valor[t=0]) - 1 (BOOST.MD)
    cyclical_features: bool = True  # Features sin/cos (BOOST.MD)
    min_sequence_length: int = 2048 # Mínimo de dados por sequência
    validation_split: float = 0.2   # 20% para validação
    dtype: str = 'float32'         # Tipo de dados para features numéricas
    
    # Assets para unificação - CORREÇÃO: configurável via YAML
    target_assets: List[str] = field(default_factory=lambda: ["BTCUSDT", "ETHUSDT", "ETHBTC", "BNBUSDT"])


class CryptoDatasetBuilder(DatasetBuilder):
    """
    Builder SOTA para dados de criptomoedas com filosofia "Tudo já está no preço"
    
    Features implementadas:
    1. Unificação: Todos os ativos em um dataset único
    2. Anonimização: Remove item_id para forçar aprendizado de padrões universais
    3. Normalização por janela: Foca na forma do padrão, não escala absoluta
    4. Features temporais cíclicas: Contexto temporal explícito
    5. Features técnicas essenciais: Apenas returns e ratios básicos
    6. Dados puros da Binance: OHLCV + volume + trades sem over-engineering
    7. Compatibilidade Moirai-MoE: Formato otimizado para MoE
    
    Filosofia SOTA:
    - Menos é mais: o modelo deve aprender padrões dos dados brutos
    - Qualidade > Quantidade: poucos features bem escolhidos
    - "Everything is in the price": não criar features redundantes
    """
    
    def __init__(
        self,
        data_path: Union[str, Path],
        config: Optional[CryptoConfig] = None
    ):
        self.data_path = Path(data_path)
        self.config = config or CryptoConfig()
        
        # Campos de dados separados por tipo
        self.numerical_fields = [
            'open', 'high', 'low', 'close', 'volume',
            'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume'
        ]
        
        self.datetime_fields = ['open_time']
        self.technical_fields: Set[str] = set()  # Será preenchido durante feature engineering
        self.cyclical_fields: Set[str] = set()   # Será preenchido durante feature engineering
        
        # Cache para dados processados
        self._processed_cache = {}
        
    def build_dataset(self, split: str = "train") -> Dataset:
        """
        Constrói dataset unificado e anônimo para o split especificado
        
        Args:
            split: "train", "validation", ou "test"
            
        Returns:
            Dataset HuggingFace formatado para uni2ts
        """
        print(f"🚀 Construindo dataset crypto SOTA - Split: {split}")
        print(f"📊 Configuração: {self.config}")
        
        # 1. Carregar dados brutos
        raw_data = self._load_raw_data()
        
        # 2. Unificar e anonimizar (BOOST.MD)
        unified_data = self._unify_and_anonymize(raw_data)
        
        # 3. Aplicar engenharia de features (BOOST.MD)
        enhanced_data = self._apply_feature_engineering(unified_data)
        
        # 4. Criar sequências de treino
        sequences = self._create_sequences(enhanced_data)
        
        # 5. Split treino/validação/teste
        train_sequences, val_sequences, test_sequences = self._split_data(sequences)
        
        # 6. Selecionar split apropriado
        if split == "train":
            selected_sequences = train_sequences
        elif split == "validation":
            selected_sequences = val_sequences
        else:  # test
            selected_sequences = test_sequences
        
        # 7. Converter para formato HuggingFace
        hf_dataset = self._create_hf_dataset(selected_sequences, split)
        
        print(f"✅ Dataset {split} criado: {len(hf_dataset)} sequências")
        return hf_dataset
    
    def _load_raw_data(self) -> Dict[str, pd.DataFrame]:
        """Carrega dados brutos de todos os ativos"""
        print("📥 Carregando dados brutos...")
        
        raw_data = {}
        dtype_map = {field: self.config.dtype for field in self.numerical_fields}
        
        for asset in self.config.target_assets:
            asset_files = list(self.data_path.glob(f"**/{asset}*.parquet"))
            
            if not asset_files:
                print(f"⚠️ Nenhum arquivo encontrado para {asset}")
                continue
                
            # Carregar arquivo mais recente
            latest_file = sorted(asset_files)[-1]
            print(f"📁 Carregando {asset}: {latest_file}")
            
            # Carregar com dtypes corretos
            df = pd.read_parquet(latest_file)
            
            # Converter campos numéricos para dtype configurado
            for field in self.numerical_fields:
                if field in df.columns:
                    df[field] = df[field].astype(self.config.dtype)
            
            # Garantir datetime
            if 'open_time' in df.columns:
                df['open_time'] = pd.to_datetime(df['open_time'])
            
            # Validar campos obrigatórios
            missing_fields = [f for f in self.numerical_fields if f not in df.columns]
            if missing_fields:
                print(f"⚠️ Campos faltando para {asset}: {missing_fields}")
                continue
            
            df = df.sort_values('open_time').reset_index(drop=True)
            raw_data[asset] = df
            print(f"   ✅ {asset}: {len(df):,} registros")
            
            # Validar tipos de dados
            print(f"   📊 Tipos de dados:")
            for col in df.columns:
                print(f"      {col}: {df[col].dtype}")
        
        return raw_data
    
    def _unify_and_anonymize(self, raw_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Unifica todos os ativos em um dataset único e anônimo (BOOST.MD)
        Remove item_id para forçar aprendizado de padrões universais
        """
        print("🔄 Unificando e anonimizando dataset (BOOST.MD)...")
        
        unified_sequences = []
        
        for asset_name, df in raw_data.items():
            print(f"   Processando {asset_name}...")
            
            # Adicionar metadados mas não como feature de treinamento
            df_copy = df.copy()
            df_copy['_original_asset'] = asset_name  # Apenas para debugging/validação
            df_copy['_sequence_id'] = f"{asset_name}_{pd.Timestamp.now().strftime('%Y%m%d')}"
            
            # Garantir tipos de dados consistentes
            for field in self.numerical_fields:
                if field in df_copy.columns:
                    df_copy[field] = df_copy[field].astype(self.config.dtype)
            
            # Validar sequência mínima
            if len(df_copy) < self.config.min_sequence_length:
                print(f"   ⚠️ {asset_name}: Sequência muito curta ({len(df_copy)} < {self.config.min_sequence_length})")
                continue
            
            unified_sequences.append(df_copy)
        
        # Concatenar tudo
        unified_df = pd.concat(unified_sequences, ignore_index=True)
        
        print(f"   ✅ Dataset unificado: {len(unified_df):,} registros de {len(raw_data)} ativos")
        return unified_df
    
    def _apply_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica engenharia de features SOTA (BOOST.MD)
        1. Normalização por janela
        2. Features cíclicas explícitas
        """
        print("🔧 Aplicando engenharia de features SOTA...")
        
        df_enhanced = df.copy()
        
        # 1. Features cíclicas explícitas (BOOST.MD)
        if self.config.cyclical_features:
            df_enhanced = self._add_cyclical_features(df_enhanced)
            
        # 2. Outras features técnicas, garantindo dtype correto
        df_enhanced = self._add_technical_features(df_enhanced)
            
        # 3. Normalização por janela (se configurado)
        if self.config.window_normalization:
            df_enhanced = self._apply_window_normalization(df_enhanced)
        
        # Validar tipos de dados após feature engineering
        print("\n📊 Validando tipos de dados após feature engineering:")
        for col in df_enhanced.columns:
            if col not in ['open_time', '_original_asset', '_sequence_id']:
                if df_enhanced[col].dtype != self.config.dtype:
                    print(f"   ⚠️ Convertendo {col} de {df_enhanced[col].dtype} para {self.config.dtype}")
                    df_enhanced[col] = df_enhanced[col].astype(self.config.dtype)
                else:
                    print(f"   ✅ {col}: {df_enhanced[col].dtype}")
        
        return df_enhanced
        
    def _add_cyclical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adiciona features temporais cíclicas explícitas (BOOST.MD)
        Retorna todas as features cíclicas com dtype correto
        """
        print("   🔄 Adicionando features cíclicas (sin/cos)...")
        
        df = df.copy()
        
        # Extrair componentes temporais
        df['_minute'] = df['open_time'].dt.minute
        df['_hour'] = df['open_time'].dt.hour
        df['_weekday'] = df['open_time'].dt.weekday
        
        # Criar features cíclicas (sempre retorna float64)
        for col, max_val in [('_minute', 60), ('_hour', 24), ('_weekday', 7)]:
            # Converter para radianos
            values_rad = 2 * np.pi * df[col] / max_val
            
            # Criar sin/cos e garantir dtype correto
            sin_col = f"{col}_sin"
            cos_col = f"{col}_cos"
            
            df[sin_col] = np.sin(values_rad).astype(self.config.dtype)
            df[cos_col] = np.cos(values_rad).astype(self.config.dtype)
            
            # Registrar como feature cíclica
            self.cyclical_fields.add(sin_col)
            self.cyclical_fields.add(cos_col)
            
            # Remover coluna temporária
            df = df.drop(columns=[col])
        
        return df
    
    def _add_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adiciona apenas features essenciais - SOTA simplificado
        Princípio: "Tudo já está no preço" - usar apenas dados da Binance
        """
        print("   📊 Adicionando features essenciais (SOTA simplificado)...")
        
        df = df.copy()
        
        # APENAS features essenciais que complementam os dados da Binance
        
        # 1. Returns simples (informação de momentum básica)
        df['price_return'] = (df['close'].pct_change()).astype(self.config.dtype)
        df['volume_return'] = (df['volume'].pct_change()).astype(self.config.dtype)
        
        # Registrar como technical features
        self.technical_fields.add('price_return')
        self.technical_fields.add('volume_return')
        
        # 2. Volume-price relationship (única relação não capturada no preço puro)
        df['volume_price_ratio'] = (df['volume'] / df['close']).astype(self.config.dtype)
        self.technical_fields.add('volume_price_ratio')
        
        # 3. Intraday range (informação de volatilidade intraday)
        df['high_low_ratio'] = ((df['high'] - df['low']) / df['close']).astype(self.config.dtype)
        self.technical_fields.add('high_low_ratio')
        
        # Tratar NaN e inf de forma simples
        df = df.replace([np.inf, -np.inf], 0)
        df = df.fillna(0)
        
        print(f"   ✅ Features essenciais criadas: {len(self.technical_fields)} campos SOTA")
        print(f"      📋 Lista: {sorted(self.technical_fields)}")
        
        return df
        
    def _apply_window_normalization(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica normalização por janela APENAS nos dados essenciais da Binance
        Princípio SOTA: normalizar apenas preços/volumes, não features derivadas
        """
        print("   📈 Aplicando normalização por janela (apenas dados Binance essenciais)...")
        
        df = df.copy()
        window_size = self.config.context_length
        
        # APENAS dados essenciais da Binance para normalização
        essential_price_fields = ['open', 'high', 'low', 'close']
        essential_volume_fields = ['volume', 'quote_asset_volume']
        
        # Normalizar preços (mais importante)
        for col in essential_price_fields:
            if col in df.columns:
                # Normalização por janela: (valor[t] / valor[t=0]) - 1
                ref_values = df[col].rolling(window_size, min_periods=1).apply(
                    lambda x: x.iloc[0] if len(x) > 0 else x.iloc[-1]
                )
                df[f"{col}_norm"] = ((df[col] / ref_values) - 1).astype(self.config.dtype)
                print(f"      ✅ Normalizado: {col} -> {col}_norm")
        
        # Normalizar volumes (menos crítico, mas útil)
        for col in essential_volume_fields:
            if col in df.columns:
                ref_values = df[col].rolling(window_size, min_periods=1).apply(
                    lambda x: x.iloc[0] if len(x) > 0 else x.iloc[-1]
                )
                df[f"{col}_norm"] = ((df[col] / ref_values) - 1).astype(self.config.dtype)
                print(f"      ✅ Normalizado: {col} -> {col}_norm")
        
        # NÃO normalizar features técnicas (já são ratios/returns normalizados)
        # NÃO normalizar features cíclicas (já são sin/cos normalizados)
        
        print(f"   ✅ Normalização SOTA aplicada apenas em dados essenciais")
        
        return df
        
    def _split_data(self, sequences: List[dict]) -> Tuple[List[dict], List[dict], List[dict]]:
        """
        Split sequences into train/validation/test sets
        Maintains temporal order and asset distribution
        """
        print(f"📊 Dividindo {len(sequences)} sequências em treino/validação/teste...")
        
        # Group sequences by asset to ensure balanced split
        asset_sequences = {}
        for seq in sequences:
            asset = seq['asset']
            if asset not in asset_sequences:
                asset_sequences[asset] = []
            asset_sequences[asset].append(seq)
        
        train_seqs, val_seqs, test_seqs = [], [], []
        
        for asset, seqs in asset_sequences.items():
            # Sort by start time to maintain temporal order
            seqs = sorted(seqs, key=lambda x: x['start_time'])
            
            # Calculate split indices
            n_seqs = len(seqs)
            val_split = int(n_seqs * (1 - self.config.validation_split - 0.1))  # 10% for test
            test_split = int(n_seqs * (1 - 0.1))
            
            # Split maintaining temporal order
            train_seqs.extend(seqs[:val_split])
            val_seqs.extend(seqs[val_split:test_split])
            test_seqs.extend(seqs[test_split:])
            
            print(f"   {asset}: {len(seqs[:val_split])}/{len(seqs[val_split:test_split])}/{len(seqs[test_split:])} (train/val/test)")
        
        print(f"✅ Split final: {len(train_seqs)}/{len(val_seqs)}/{len(test_seqs)} sequências")
        return train_seqs, val_seqs, test_seqs
        
    def _create_sequences(self, df: pd.DataFrame) -> List[dict]:
        """
        Cria sequências de treino com tipos de dados consistentes
        Otimizado para performance e memory usage
        """
        print("🔄 Criando sequências de treinamento...")
        
        sequences = []
        total_len = len(df)
        step_size = self.config.prediction_length
        
        # Definir feature columns com prioridade explícita
        feature_cols = self._get_feature_columns(df)
        
        print(f"   📊 Features selecionadas ({len(feature_cols)}): {feature_cols[:5]}...")
        
        # Pre-process features matrix for efficiency
        try:
            features_matrix = df[feature_cols].to_numpy(dtype=self.config.dtype)
            print(f"   ✅ Features matrix shape: {features_matrix.shape}")
        except Exception as e:
            print(f"   ❌ Erro criando features matrix: {e}")
            # Fallback: force dtype conversion
            features_matrix = df[feature_cols].astype(self.config.dtype).to_numpy()
            
        # Extract metadata columns
        sequence_ids = df['_sequence_id'].values
        assets = df['_original_asset'].values
        timestamps = df['open_time'].values
        
        # Create sequences with vectorized operations
        min_sequence_length = self.config.context_length + self.config.prediction_length
        
        for start_idx in range(0, total_len - min_sequence_length + 1, step_size):
            end_idx = start_idx + min_sequence_length
            
            # Extract sequence efficiently
            sequence_features = features_matrix[start_idx:end_idx]
            
            # Split into context and target
            context_features = sequence_features[:self.config.context_length]
            target_features = sequence_features[self.config.context_length:]
            
            # Create sequence dictionary
            sequence = {
                "target": context_features[:, 0:1],  # Use first feature column as target (close price)
                "feat_dynamic_real": context_features,
                "feat_static_cat": [0],  # Anonymous training - no asset ID
                "sequence_id": sequence_ids[start_idx],
                "asset": assets[start_idx] if not self.config.anonymous_training else "unified",
                "start_time": timestamps[start_idx].isoformat(),
                "end_time": timestamps[end_idx-1].isoformat()
            }
            
            sequences.append(sequence)
        
        print(f"   ✅ Criadas {len(sequences)} sequências")
        return sequences
        
    def _get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Retorna lista ordenada de features SOTA simplificadas
        Foco em dados essenciais da Binance + contexto temporal
        """
        feature_list = []
        
        # 1. Dados essenciais da Binance (normalizados se ativo)
        essential_fields = ['open', 'high', 'low', 'close', 'volume', 'quote_asset_volume']
        
        for field in essential_fields:
            if self.config.window_normalization and f"{field}_norm" in df.columns:
                feature_list.append(f"{field}_norm")
            elif field in df.columns:
                feature_list.append(field)
        
        # 2. Features técnicas essenciais (apenas as 4 criadas)
        essential_technical = ['price_return', 'volume_return', 'volume_price_ratio', 'high_low_ratio']
        for field in essential_technical:
            if field in df.columns:
                feature_list.append(field)
        
        # 3. Features temporais cíclicas (contexto temporal)
        cyclical_features = sorted([col for col in self.cyclical_fields if col in df.columns])
        feature_list.extend(cyclical_features)
        
        # 4. Adicionar outros campos numéricos da Binance se existirem
        other_binance_fields = ['number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume']
        for field in other_binance_fields:
            if field in df.columns:
                if self.config.window_normalization and f"{field}_norm" in df.columns:
                    feature_list.append(f"{field}_norm")
                else:
                    feature_list.append(field)
        
        # Validar existência
        existing_features = [col for col in feature_list if col in df.columns]
        missing_features = set(feature_list) - set(existing_features)
        
        if missing_features:
            print(f"   ⚠️ Features esperadas mas não encontradas: {missing_features}")
        
        print(f"   📊 Features SOTA selecionadas: {len(existing_features)}")
        print(f"      💰 Dados Binance: {[f for f in existing_features if any(x in f for x in essential_fields)]}")
        print(f"      📈 Features técnicas: {[f for f in existing_features if f in essential_technical]}")
        print(f"      🕐 Features temporais: {[f for f in existing_features if '_sin' in f or '_cos' in f]}")
        
        return existing_features
        
    def _create_hf_dataset(self, sequences: List[dict], split: str) -> Dataset:
        """
        Converte sequências para formato uni2ts compatível
        Usa schema otimizado para time series forecasting
        """
        print(f"🏗️ Criando dataset HuggingFace para {split}...")
        
        # Converter para formato uni2ts padrão
        formatted_data = []
        
        for seq in sequences:
            # Formato compatível com uni2ts DataLoader
            formatted_seq = {
                "target": seq["target"].flatten().tolist(),  # Time series values
                "feat_dynamic_real": seq["feat_dynamic_real"].tolist(),  # Dynamic features
                "feat_static_cat": seq["feat_static_cat"],  # Static categorical features
                "start": seq["start_time"],  # Start timestamp
                "item_id": seq["sequence_id"] if not self.config.anonymous_training else f"unified_{len(formatted_data)}"
            }
            formatted_data.append(formatted_seq)
        
        # Schema otimizado para uni2ts
        features_schema = Features({
            "target": Sequence(Value("float32")),
            "feat_dynamic_real": Sequence(Sequence(Value("float32"))),
            "feat_static_cat": Sequence(Value("int64")),
            "start": Value("string"),
            "item_id": Value("string")
        })
        
        # Criar dataset com cache otimizado
        hf_dataset = Dataset.from_list(
            formatted_data,
            features=features_schema
        )
        
        print(f"   ✅ Dataset {split}: {len(hf_dataset)} sequências, {hf_dataset.num_columns} colunas")
        return hf_dataset
    
    def get_feature_info(self) -> Dict:
        """Retorna informações detalhadas sobre features para debugging/monitoring"""
        return {
            'numerical_fields': self.numerical_fields,
            'technical_fields': sorted(list(self.technical_fields)),
            'cyclical_fields': sorted(list(self.cyclical_fields)),
            'total_features': len(self.numerical_fields) + len(self.technical_fields) + len(self.cyclical_fields),
            'normalization': 'window_based' if self.config.window_normalization else 'none',
            'context_length': self.config.context_length,
            'prediction_length': self.config.prediction_length,
            'anonymous_training': self.config.anonymous_training,
            'dtype': self.config.dtype,
            'unified_dataset': self.config.unified_dataset,
            'target_assets': self.config.target_assets
        }

    def load_dataset(self, transform_map: dict = None) -> Dataset:
        """
        Implementação padrão uni2ts da classe base DatasetBuilder
        
        Args:
            transform_map: Mapa de transformações (opcional, uni2ts padrão)
            
        Returns:
            Dataset de treinamento pronto para DataLoader
        """
        return self.build_dataset(split="train")
    
    def validate_config(self, check_data_path: bool = True) -> bool:
        """
        Valida configuração antes do build
        
        Args:
            check_data_path: Se deve validar existência do data_path (False para testes)
        """
        try:
            # Validar data_path existe (opcional para testes)
            if check_data_path and not self.data_path.exists():
                raise ValueError(f"Data path não encontrado: {self.data_path}")
            elif not check_data_path:
                print(f"⚠️ Pulando validação de data_path (modo teste)")
            
            # Validar target_assets não vazio
            if not self.config.target_assets:
                raise ValueError("Lista target_assets não pode estar vazia")
            
            # Validar context_length vs prediction_length
            if self.config.context_length < self.config.prediction_length:
                print(f"⚠️ Context length ({self.config.context_length}) < prediction length ({self.config.prediction_length})")
            
            # Validar numerical fields
            if not self.numerical_fields:
                raise ValueError("Nenhum campo numérico definido")
            
            print("✅ Configuração validada com sucesso")
            return True
            
        except Exception as e:
            print(f"❌ Erro na validação: {e}")
            return False
    
    def build_datasets(self) -> Tuple[Dataset, Dataset, Dataset]:
        """
        Constrói e retorna os três datasets: treino, validação e teste.
        
        Returns:
            Tuple contendo (train_dataset, val_dataset, test_dataset)
        """
        print("🏗️ Construindo datasets de treino, validação e teste...")
        
        train_dataset = self.build_dataset(split="train")
        val_dataset = self.build_dataset(split="validation") 
        test_dataset = self.build_dataset(split="test")
        
        print(f"✅ Datasets construídos:")
        print(f"   🚂 Treino: {len(train_dataset)} amostras")
        print(f"   🔬 Validação: {len(val_dataset)} amostras")
        print(f"   🧪 Teste: {len(test_dataset)} amostras")
        
        return train_dataset, val_dataset, test_dataset
