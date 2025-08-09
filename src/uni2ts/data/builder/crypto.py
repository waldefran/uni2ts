"""
Crypto Dataset Builder SOTA - Dataset Unificado e Anônimo
Implementa todas as melhorias do BOOST.MD para estado da arte

UPGRADE BOOST.MD implementado:
- Dataset unificado: todos os ativos em um só dataset
- Anonimização: remove item_id durante treinamento
- Normalização por janela: (valor[t] / valor[t=0]) - 1
- Features cíclicas explícitas: sin/cos para minute, hour, weekday
- Compatível com Moirai-MoE architecture
"""

import pandas as pd
import numpy as np
import torch
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
from datetime import datetime
import pickle
from datasets import Dataset, Features, Value, Sequence
from dataclasses import dataclass

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
    
    # Assets para unificação
    target_assets: List[str] = None
    
    def __post_init__(self):
        if self.target_assets is None:
            self.target_assets = ["BTCUSDT", "ETHUSDT", "ETHBTC", "BNBUSDT"]


class CryptoDatasetBuilder(DatasetBuilder):
    """
    Builder SOTA para dados de criptomoedas com todas as melhorias do BOOST.MD
    
    Features implementadas:
    1. Unificação: Todos os ativos em um dataset único
    2. Anonimização: Remove item_id para forçar aprendizado de padrões universais
    3. Normalização por janela: Foca na forma do padrão, não escala absoluta
    4. Features cíclicas: Contexto temporal explícito
    5. Compatibilidade Moirai-MoE: Formato otimizado para MoE
    """
    
    def __init__(
        self,
        data_path: Union[str, Path],
        config: Optional[CryptoConfig] = None
    ):
        self.data_path = Path(data_path)
        self.config = config or CryptoConfig()
        
        # Campos de dados de candles
        self.candle_fields = [
            'open', 'high', 'low', 'close', 'volume',
            'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume'
        ]
        
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
        
        # 5. Split treino/validação
        train_sequences, val_sequences = self._split_data(sequences)
        
        # 6. Selecionar split apropriado
        selected_sequences = train_sequences if split == "train" else val_sequences
        
        # 7. Converter para formato HuggingFace
        hf_dataset = self._create_hf_dataset(selected_sequences, split)
        
        print(f"✅ Dataset {split} criado: {len(hf_dataset)} sequências")
        return hf_dataset
    
    def _load_raw_data(self) -> Dict[str, pd.DataFrame]:
        """Carrega dados brutos de todos os ativos"""
        print("📥 Carregando dados brutos...")
        
        raw_data = {}
        
        for asset in self.config.target_assets:
            asset_files = list(self.data_path.glob(f"**/{asset}*.parquet"))
            
            if not asset_files:
                print(f"⚠️ Nenhum arquivo encontrado para {asset}")
                continue
                
            # Carregar arquivo mais recente
            latest_file = sorted(asset_files)[-1]
            print(f"📁 Carregando {asset}: {latest_file}")
            
            df = pd.read_parquet(latest_file)
            
            # Validar campos obrigatórios
            missing_fields = [f for f in self.candle_fields if f not in df.columns]
            if missing_fields:
                print(f"⚠️ Campos faltando para {asset}: {missing_fields}")
                continue
            
            # Garantir timestamp como datetime
            if 'open_time' in df.columns:
                df['open_time'] = pd.to_datetime(df['open_time'])
                df = df.sort_values('open_time').reset_index(drop=True)
            
            raw_data[asset] = df
            print(f"   ✅ {asset}: {len(df):,} registros")
        
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
        
        # 2. Outras features derivadas
        df_enhanced = self._add_technical_features(df_enhanced)
        
        return df_enhanced
    
    def _add_cyclical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adiciona features temporais cíclicas explícitas (BOOST.MD)"""
        print("   🔄 Adicionando features cíclicas (sin/cos)...")
        
        if 'open_time' not in df.columns:
            print("   ⚠️ Coluna open_time não encontrada")
            return df
        
        # Extrair componentes temporais
        df['minute_of_hour'] = df['open_time'].dt.minute
        df['hour_of_day'] = df['open_time'].dt.hour
        df['day_of_week'] = df['open_time'].dt.dayofweek
        
        # Transformações cíclicas sin/cos
        df['minute_sin'] = np.sin(2 * np.pi * df['minute_of_hour'] / 60)
        df['minute_cos'] = np.cos(2 * np.pi * df['minute_of_hour'] / 60)
        
        df['hour_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / 24)
        
        df['weekday_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['weekday_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # Remover features temporais originais (mantém apenas sin/cos)
        df = df.drop(['minute_of_hour', 'hour_of_day', 'day_of_week'], axis=1)
        
        print("   ✅ Features cíclicas adicionadas: minute, hour, weekday (sin/cos)")
        return df
    
    def _add_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adiciona features técnicas básicas"""
        print("   📈 Adicionando features técnicas...")
        
        # Returns
        df['price_return'] = df['close'].pct_change()
        df['volume_return'] = df['volume'].pct_change()
        
        # Volatilidade (rolling std dos returns)
        df['volatility_5m'] = df['price_return'].rolling(5).std()
        df['volatility_15m'] = df['price_return'].rolling(15).std()
        
        # VWAP (Volume Weighted Average Price)
        df['vwap'] = (df['close'] * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
        
        # Spread relativo
        df['spread_ratio'] = (df['high'] - df['low']) / df['close']
        
        # Volume ratio
        df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        
        return df
    
    def _create_sequences(self, df: pd.DataFrame) -> List[Dict]:
        """
        Cria sequências de contexto + predição com normalização por janela (BOOST.MD)
        """
        print("📦 Criando sequências com normalização por janela...")
        
        sequences = []
        context_len = self.config.context_length
        pred_len = self.config.prediction_length
        
        # Campos que serão normalizados por janela
        price_volume_fields = ['open', 'high', 'low', 'close', 'volume', 
                              'quote_asset_volume', 'taker_buy_base_asset_volume', 
                              'taker_buy_quote_asset_volume']
        
        # Campos que não são normalizados (já são relativos ou cíclicos)
        static_fields = [col for col in df.columns 
                        if col not in price_volume_fields 
                        and not col.startswith('_')
                        and col != 'open_time']
        
        # Agrupar por sequência de ativo (se houver quebras temporais grandes)
        sequence_groups = self._group_sequences(df)
        
        for group_df in sequence_groups:
            group_len = len(group_df)
            
            # Criar sequências deslizantes
            for i in range(0, group_len - context_len - pred_len + 1, pred_len):
                context_end = i + context_len
                pred_end = context_end + pred_len
                
                # Extrair janela de contexto + predição
                window_df = group_df.iloc[i:pred_end].copy()
                
                # Aplicar normalização por janela (BOOST.MD)
                if self.config.window_normalization:
                    window_df = self._apply_window_normalization(window_df, price_volume_fields)
                
                # Separar contexto e target
                context_data = window_df.iloc[:context_len]
                target_data = window_df.iloc[context_len:]
                
                # Construir features finais
                feature_matrix = self._build_feature_matrix(context_data, price_volume_fields, static_fields)
                target_vector = target_data['close'].values  # Predizer close price
                
                # Metadados (não usados em treinamento anônimo)
                metadata = {
                    'original_asset': window_df['_original_asset'].iloc[0] if '_original_asset' in window_df else 'unknown',
                    'sequence_id': window_df['_sequence_id'].iloc[0] if '_sequence_id' in window_df else f'seq_{len(sequences)}',
                    'start_time': window_df['open_time'].iloc[0] if 'open_time' in window_df else None,
                    'end_time': window_df['open_time'].iloc[-1] if 'open_time' in window_df else None
                }
                
                sequences.append({
                    'target': feature_matrix,  # [context_len, n_features] - input do modelo
                    'future_target': target_vector,  # [pred_len] - ground truth
                    'start': context_data['open_time'].iloc[0] if 'open_time' in context_data else pd.Timestamp.now(),
                    'freq': '1min',
                    'metadata': metadata
                })
        
        print(f"   ✅ Criadas {len(sequences)} sequências")
        return sequences
    
    def _group_sequences(self, df: pd.DataFrame) -> List[pd.DataFrame]:
        """Agrupa dados em sequências contínuas por ativo"""
        if '_original_asset' not in df.columns:
            return [df]
        
        groups = []
        for asset in df['_original_asset'].unique():
            asset_df = df[df['_original_asset'] == asset].copy()
            
            # Detectar quebras temporais (gaps > 5 minutos)
            if 'open_time' in asset_df.columns:
                time_diffs = asset_df['open_time'].diff()
                gap_mask = time_diffs > pd.Timedelta(minutes=5)
                gap_indices = asset_df.index[gap_mask].tolist()
                
                # Dividir em grupos contínuos
                start_idx = 0
                for gap_idx in gap_indices + [len(asset_df)]:
                    if gap_idx - start_idx >= self.config.min_sequence_length:
                        groups.append(asset_df.iloc[start_idx:gap_idx])
                    start_idx = gap_idx
            else:
                groups.append(asset_df)
        
        return groups
    
    def _apply_window_normalization(
        self, 
        window_df: pd.DataFrame, 
        price_volume_fields: List[str]
    ) -> pd.DataFrame:
        """
        Aplica normalização por janela: (valor[t] / valor[t=0]) - 1 (BOOST.MD)
        Foca o modelo na forma do padrão, não na escala absoluta
        """
        normalized_df = window_df.copy()
        
        for field in price_volume_fields:
            if field in normalized_df.columns:
                values = normalized_df[field].values
                
                # Evitar divisão por zero
                first_value = values[0]
                if first_value != 0 and not np.isnan(first_value):
                    normalized_values = (values / first_value) - 1
                    normalized_df[field] = normalized_values
                else:
                    # Se primeiro valor é 0 ou NaN, usar diferenças percentuais
                    normalized_df[field] = normalized_df[field].pct_change().fillna(0)
        
        return normalized_df
    
    def _build_feature_matrix(
        self, 
        context_data: pd.DataFrame,
        price_volume_fields: List[str],
        static_fields: List[str]
    ) -> np.ndarray:
        """Constrói matriz de features para o modelo"""
        
        feature_columns = []
        
        # 1. Campos de preço/volume (normalizados)
        for field in price_volume_fields:
            if field in context_data.columns:
                feature_columns.append(context_data[field].values)
        
        # 2. Features estáticas (cíclicas, técnicas)
        for field in static_fields:
            if field in context_data.columns:
                feature_columns.append(context_data[field].values)
        
        # Empilhar features
        if feature_columns:
            feature_matrix = np.column_stack(feature_columns)
        else:
            # Fallback: apenas close price normalizado
            feature_matrix = context_data['close'].values.reshape(-1, 1)
        
        # Tratar NaNs
        feature_matrix = np.nan_to_num(feature_matrix, nan=0.0)
        
        return feature_matrix.astype(np.float32)
    
    def _split_data(self, sequences: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """Split treino/validação temporal"""
        
        # Ordenar por timestamp se disponível
        sequences_with_time = [(seq, seq['start']) for seq in sequences if seq['start'] is not None]
        sequences_without_time = [seq for seq in sequences if seq['start'] is None]
        
        if sequences_with_time:
            # Split temporal: últimos 20% para validação
            sequences_with_time.sort(key=lambda x: x[1])
            split_point = int(len(sequences_with_time) * (1 - self.config.validation_split))
            
            train_sequences = [seq for seq, _ in sequences_with_time[:split_point]]
            val_sequences = [seq for seq, _ in sequences_with_time[split_point:]]
        else:
            # Split aleatório se não houver timestamps
            np.random.shuffle(sequences)
            split_point = int(len(sequences) * (1 - self.config.validation_split))
            train_sequences = sequences[:split_point]
            val_sequences = sequences[split_point:]
        
        # Adicionar sequências sem timestamp ao treino
        train_sequences.extend(sequences_without_time)
        
        return train_sequences, val_sequences
    
    def _create_hf_dataset(self, sequences: List[Dict], split: str) -> Dataset:
        """Converte sequências para formato HuggingFace Dataset"""
        
        if not sequences:
            raise ValueError(f"Nenhuma sequência disponível para split {split}")
        
        # Extrair dados
        targets = [seq['target'] for seq in sequences]
        future_targets = [seq['future_target'] for seq in sequences]
        starts = [seq['start'] for seq in sequences]
        freqs = [seq['freq'] for seq in sequences]
        
        # Schema HuggingFace
        n_features = targets[0].shape[1] if len(targets[0].shape) > 1 else 1
        
        features = Features({
            'target': Sequence(Sequence(Value('float32'))),  # [context_len, n_features]
            'future_target': Sequence(Value('float32')),     # [pred_len]
            'start': Value('timestamp[s]'),
            'freq': Value('string'),
            # NOTA: item_id removido intencionalmente para anonimização (BOOST.MD)
        })
        
        # Converter para formato correto
        dataset_dict = {
            'target': [target.tolist() for target in targets],
            'future_target': [future.tolist() for future in future_targets],
            'start': [int(start.timestamp()) if start else 0 for start in starts],
            'freq': freqs
        }
        
        # Criar dataset
        dataset = Dataset.from_dict(dataset_dict, features=features)
        
        return dataset
    
    def get_feature_info(self) -> Dict:
        """Retorna informações sobre features construídas"""
        info = {
            'price_volume_fields': self.candle_fields,
            'cyclical_features': ['minute_sin', 'minute_cos', 'hour_sin', 'hour_cos', 'weekday_sin', 'weekday_cos'],
            'technical_features': ['price_return', 'volume_return', 'volatility_5m', 'volatility_15m', 'vwap', 'spread_ratio', 'volume_ratio'],
            'normalization': 'window_based' if self.config.window_normalization else 'none',
            'context_length': self.config.context_length,
            'prediction_length': self.config.prediction_length,
            'anonymous_training': self.config.anonymous_training
        }
        return info
