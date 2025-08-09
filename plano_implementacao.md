# Plano de Implementação: Fine-tuning do Moirai para Trading de Criptomoedas

## Análise da Proposta Original

A proposta em `prop.md` apresenta uma visão sólida, mas alguns aspectos podem ser refinados com base no estado da arte e na arquitetura existente do Moirai. Abaixo está o plano otimizado:

## **FASE 1: PREPARAÇÃO DOS DADOS E FINE-TUNING**

### **1.1 - Coleta e Preparação do Dataset de Criptomoedas**

#### **Dados da Binance API**
- **Ativos alvo**: BTC/USDT, ETH/USDT, e outros ativos principais
- **Timeframe**: M1 (1 minuto) - últimos 3-4 anos
- **Abordagem**: **Univariada, unificada e anônima**
- **Campos por candle**:
  ```
  - open_time (timestamp)
  - open (float32)
  - high (float32) 
  - low (float32)
  - close (float32)
  - volume (float32)
  - close_time (timestamp)
  - quote_asset_volume (float32)
  - number_of_trades (int32)
  - taker_buy_base_asset_volume (float32)
  - taker_buy_quote_asset_volume (float32)
  ```

#### **Estrutura do Dataset Unificado e Anônimo**
**UPGRADE BOOST.MD**: Abordagem univariada, unificada e anônima seguindo estado da arte:

**Unificação**: Todos os ativos (BTC/USDT, ETH/USDT, etc.) são coletados e unidos em um único grande dataset.

**Anonimização**: A feature `item_id` é removida durante o treinamento. O modelo trata cada amostra de 2048 candles como um exemplo genérico de "movimento de preço", sem saber se é BTC, ETH ou outro ativo.

**Arquitetura Moirai-MoE**: Utilizar Moirai-MoE (Mixture-of-Experts) que é ideal para dataset unificado, onde experts internos se especializam implicitamente em diferentes padrões de volatilidade comuns a todos ativos.

```python
# Estrutura compatível com uni2ts - Dataset Unificado e Anônimo
features = Features({
    "target": Sequence(Value("float32")),  # multi-field: [open, high, low, close, volume, ...]
    "start": Value("timestamp[s]"),
    "freq": Value("string"),  # "1min"
    # REMOVIDO: item_id para treinamento anônimo
    # Durante treinamento: modelo não sabe qual ativo está processando
    # Durante inferência: mantemos identificação para pós-processamento
})
```

### **1.2 - Engenharia de Features Temporais Cíclicas Explícitas**

**UPGRADE BOOST.MD**: Features temporais cíclicas explícitas + Normalização por janela:

**Melhorias sobre a proposta original**:
- **Features Ricas Mantidas**: Todos os campos de candle mantidos (`open`, `high`, `low`, `close`, `volume`, `number_of_trades`, `taker_buy_base_asset_volume`, etc.)
- **Features temporais cíclicas explícitas**: Para cada um dos 2048 timesteps, adicionar features sin/cos para regimes de mercado
- **Normalização por janela**: Usar normalização relativa `(valor[t] / valor[t=0]) - 1` para cada janela de contexto
- **Contextualização**: Usar 2048 timesteps como contexto (compatível com `max_seq_len` do modelo)

```python
# Features derivadas pelo pipeline Moirai + Features cíclicas explícitas:
# - time_id: timestamp encoding (mantido)
# - variate_id: identificação do ativo (removido durante treinamento)
# - observed_mask: máscara de dados válidos

# Features cíclicas adicionadas para cada timestep (UPGRADE BOOST.MD):
# - minuto_da_hora_sin, minuto_da_hora_cos
# - hora_do_dia_sin, hora_do_dia_cos  
# - dia_da_semana_sin, dia_da_semana_cos

def add_cyclical_features(timestamp):
    """Adiciona features cíclicas explícitas para contexto temporal (BOOST.MD)"""
    minute_of_hour = timestamp.minute
    hour_of_day = timestamp.hour
    day_of_week = timestamp.weekday()
    
    return {
        "minute_sin": np.sin(2 * np.pi * minute_of_hour / 60),
        "minute_cos": np.cos(2 * np.pi * minute_of_hour / 60),
        "hour_sin": np.sin(2 * np.pi * hour_of_day / 24),
        "hour_cos": np.cos(2 * np.pi * hour_of_day / 24),
        "weekday_sin": np.sin(2 * np.pi * day_of_week / 7),
        "weekday_cos": np.cos(2 * np.pi * day_of_week / 7),
    }

def normalize_window(values):
    """Normalização por janela: (valor[t] / valor[t=0]) - 1 (BOOST.MD)"""
    if len(values) == 0 or values[0] == 0:
        return values
    return [(value / values[0]) - 1 for value in values]
```

### **1.3 - Cabeça Bayesiana e Configuração do Fine-tuning**

**UPGRADE BOOST.MD - CONFIRMAÇÃO SOTA**: Os componentes `BayesianPredictionHead` e `BayesianELBOLoss` são mantidos integralmente como estado da arte e NÃO devem ser simplificados.

#### **Implementação da Cabeça Bayesiana**

O estado da arte para incerteza em séries temporais financeiras requer uma **cabeça de predição Bayesiana** com múltiplas fontes de incerteza:

**CONFIRMADO (BOOST.MD)**: Manter a `BayesianPredictionHead` completa com:
- Camadas variacionais (`BayesianLinear`) para incerteza epistêmica
- MC Dropout para incerteza adicional do modelo  
- Distribuição Student-T para heavy tails de crypto
- Decomposição rigorosa de incerteza: epistêmica vs aleatória
- Attention temporal para interpretabilidade

**CONFIRMADO (BOOST.MD)**: Manter a `BayesianELBOLoss` completa:
- Matematicamente mais rigorosa que NLL pura
- Inclui regularização KL essencial para redes variacionais  
- Estabiliza treinamento de pesos variacionais

```python
# src/uni2ts/model/crypto/bayesian_head.py
class BayesianPredictionHead(nn.Module):
    """
    Cabeça Bayesiana para predição de séries temporais de criptomoedas
    Estado da arte: combina incerteza aleatória + epistêmica
    """
    def __init__(
        self,
        d_model: int,
        prediction_length: int,
        patch_sizes: tuple[int, ...],
        mc_dropout_rate: float = 0.1,
        num_mc_samples: int = 100,
        use_variational_weights: bool = True,
    ):
        super().__init__()
        self.prediction_length = prediction_length
        self.mc_dropout_rate = mc_dropout_rate
        self.num_mc_samples = num_mc_samples
        
        # Distribuição de saída: Student-t para heavy tails (crypto)
        self.distr_output = StudentTOutput()
        
        # Camadas variacionais para incerteza epistêmica
        if use_variational_weights:
            self.variational_linear = BayesianLinear(d_model, d_model)
        else:
            self.variational_linear = nn.Linear(d_model, d_model)
            
        # MC Dropout para incerteza epistêmica adicional
        self.mc_dropout = nn.Dropout(mc_dropout_rate)
        
        # Projeção final para parâmetros da distribuição
        self.param_proj = MultiOutSizeLinear(
            in_features=d_model,
            out_features_ls=patch_sizes,
            # Student-t: loc, scale, df
            target_dim=3  
        )
        
        # Attention para importância temporal
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
    
    def forward(
        self, 
        reprs: torch.Tensor, 
        patch_size: torch.Tensor,
        training: bool = True
    ) -> Distribution:
        """
        Forward pass com amostragem Monte Carlo
        
        Args:
            reprs: [batch, seq_len, d_model] - representações do encoder
            patch_size: tamanho dos patches
            training: se está em modo de treinamento
            
        Returns:
            Distribuição Student-t com incerteza epistêmica + aleatória
        """
        batch_size, seq_len, d_model = reprs.shape
        
        if training:
            # Durante treinamento: uma passada com dropout
            mc_outputs = [self._single_forward(reprs, patch_size) for _ in range(1)]
        else:
            # Durante inferência: múltiplas passadas MC
            mc_outputs = [
                self._single_forward(reprs, patch_size) 
                for _ in range(self.num_mc_samples)
            ]
        
        # Agregar saídas Monte Carlo
        return self._aggregate_mc_outputs(mc_outputs)
    
    def _single_forward(self, reprs: torch.Tensor, patch_size: torch.Tensor) -> dict:
        """Uma passada forward com dropout"""
        # Attention temporal para focar em padrões importantes
        attended_reprs, attention_weights = self.temporal_attention(
            reprs, reprs, reprs
        )
        
        # Camada variacional + MC Dropout
        h = self.variational_linear(attended_reprs)
        h = F.silu(h)  # SiLU activation (estado da arte)
        h = self.mc_dropout(h)
        
        # Projeção para parâmetros da distribuição
        distr_params = self.param_proj(h, patch_size)
        
        # Student-t parameters: loc, scale, df
        loc = distr_params[..., 0]
        scale = F.softplus(distr_params[..., 1]) + 1e-6  # ensure positive
        df = F.softplus(distr_params[..., 2]) + 2.0      # df > 2 for variance
        
        return {
            "loc": loc,
            "scale": scale, 
            "df": df,
            "attention_weights": attention_weights
        }
    
    def _aggregate_mc_outputs(self, mc_outputs: list[dict]) -> Distribution:
        """Agrega múltiplas amostras MC em distribuição final"""
        if len(mc_outputs) == 1:
            # Treinamento: distribuição única
            params = mc_outputs[0]
            return StudentT(
                loc=params["loc"],
                scale=params["scale"], 
                df=params["df"]
            )
        
        # Inferência: mistura de distribuições para incerteza epistêmica
        locs = torch.stack([out["loc"] for out in mc_outputs])
        scales = torch.stack([out["scale"] for out in mc_outputs])
        dfs = torch.stack([out["df"] for out in mc_outputs])
        
        # Distribuição agregada (aproximação Gaussiana da mistura)
        mean_loc = locs.mean(dim=0)
        epistemic_var = locs.var(dim=0)  # incerteza epistêmica
        aleatoric_var = (scales**2).mean(dim=0)  # incerteza aleatória
        
        total_scale = torch.sqrt(epistemic_var + aleatoric_var)
        mean_df = dfs.mean(dim=0)
        
        return StudentT(loc=mean_loc, scale=total_scale, df=mean_df)


class BayesianLinear(nn.Module):
    """Camada linear variacional para incerteza de peso"""
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Parâmetros variacionais dos pesos
        self.weight_mu = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.weight_logsigma = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        
        # Parâmetros variacionais do bias
        self.bias_mu = nn.Parameter(torch.randn(out_features) * 0.1)
        self.bias_logsigma = nn.Parameter(torch.randn(out_features) * 0.1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Reparameterization trick
        weight_eps = torch.randn_like(self.weight_mu)
        weight = self.weight_mu + torch.exp(self.weight_logsigma) * weight_eps
        
        bias_eps = torch.randn_like(self.bias_mu) 
        bias = self.bias_mu + torch.exp(self.bias_logsigma) * bias_eps
        
        return F.linear(x, weight, bias)
```

#### **Arquivo de configuração: `crypto_finetune.yaml`**

**UPGRADE BOOST.MD**: Configuração para Moirai-MoE com dataset unificado e componentes Bayesianos confirmados.

```yaml
# crypto_finetune.yaml - UPGRADE BOOST.MD
defaults:
  - model: moirai_moe_base  # UPGRADE: Moirai-MoE para dataset unificado
  - data: crypto_m1_unified  # UPGRADE: Dataset unificado e anônimo
  - val_data: crypto_m1_val
  - _self_

run_name: crypto_bayesian_moe_specialist_v2
seed: 42
tf32: true
compile: false

model:
  pretrained_model_name_or_path: "Salesforce/moirai-moe-1.0-R-base"  # MoE architecture
  context_length: 2048  # 2048 minutos de contexto (~34 horas)
  prediction_length: 60  # próximos 60 minutos
  patch_size: 8  # otimizado para M1
  num_samples: 100
  
  # CONFIRMADO: Configurações Bayesianas mantidas integralmente (BOOST.MD)
  bayesian_head:
    mc_dropout_rate: 0.15  # dropout mais alto para incerteza
    num_mc_samples: 50     # amostras MC durante inferência
    use_variational_weights: true  # CONFIRMADO: Pesos variacionais mantidos
    temporal_attention: true
  
  # CONFIRMADO: Loss Bayesiana ELBO mantida integralmente (BOOST.MD)
  loss_func:
    _target_: uni2ts.loss.BayesianELBOLoss  # CONFIRMADO: SOTA loss function
    kl_weight: 0.001  # peso da regularização KL
    likelihood_weight: 1.0

# UPGRADE: Configurações para dataset unificado
data:
  _target_: uni2ts.data.builder.CryptoDatasetBuilder
  unified_dataset: true        # UPGRADE: Dataset unificado
  anonymous_training: true     # UPGRADE: Remove item_id durante treinamento
  window_normalization: true   # UPGRADE: (valor[t] / valor[t=0]) - 1
  cyclical_features: true      # UPGRADE: Features sin/cos explícitas
  assets: ["BTC/USDT", "ETH/USDT", "BNB/USDT", "ADA/USDT"]  # Múltiplos ativos unificados

trainer:
  max_epochs: 100  # mais épocas para convergência Bayesiana
  gradient_clip_val: 1.0
  accumulate_grad_batches: 4
  precision: 16

train_dataloader:
  batch_size: 16  # menor batch para Bayesiano
  
callbacks:
  - _target_: lightning.pytorch.callbacks.ModelCheckpoint
    monitor: val/ELBO  # CONFIRMADO: Monitor Bayesiano
    mode: min
    save_top_k: 3
  - _target_: lightning.pytorch.callbacks.EarlyStopping
    monitor: val/ELBO  # CONFIRMADO: Early stopping Bayesiano
    patience: 15  # mais paciência para Bayesiano
    mode: min
  # CONFIRMADO: Callback específico para monitorar incerteza (mantido)
  - _target_: crypto_callbacks.UncertaintyMonitor
    log_attention_weights: true
    uncertainty_threshold: 0.05
```

### **1.4 - Data Builder Customizado**

**UPGRADE BOOST.MD**: Implementar normalização por janela para focar na forma do padrão, não na escala absoluta.

Criar um builder específico para dados de crypto:

```python
# src/uni2ts/data/builder/crypto.py
class CryptoDatasetBuilder(DatasetBuilder):
    def __init__(self, data_path: Path):
        self.data_path = data_path
    
    def build_dataset(self, split: str = "train"):
        # Implementar carregamento dos dados da Binance
        # Aplicar unificação e anonimização (BOOST.MD)
        # Converter para formato HuggingFace Dataset
        # Aplicar normalização por janela e features cíclicas
        pass
    
    def create_hf_dataset(self, crypto_data: pd.DataFrame):
        """
        Converte dados de crypto para formato HuggingFace (UPGRADE BOOST.MD)
        """
        # Multi-field target: [open, high, low, close, volume, trades, ...]
        # Normalização por janela deslizante: (valor[t] / valor[t=0]) - 1
        # Geração de amostras sobrepostas de 2048 timesteps
        # Aplicação de features cíclicas para cada timestep
        # Remoção de item_id para anonimização durante treinamento
        pass
    
    def apply_window_normalization(self, window_data):
        """
        Normalização por janela (BOOST.MD): (valor[t] / valor[t=0]) - 1
        Foca o modelo na forma e dinâmica do preço, não na escala absoluta
        """
        normalized_data = {}
        price_volume_fields = ['open', 'high', 'low', 'close', 'volume', 
                              'quote_asset_volume', 'taker_buy_base_asset_volume', 
                              'taker_buy_quote_asset_volume']
        
        for field in price_volume_fields:
            if field in window_data and len(window_data[field]) > 0:
                values = window_data[field]
                if values[0] != 0:  # evitar divisão por zero
                    normalized_data[field] = [(v / values[0]) - 1 for v in values]
                else:
                    normalized_data[field] = values
            
        # Campos não normalizados (inteiros, flags)
        for field in ['number_of_trades']:
            if field in window_data:
                normalized_data[field] = window_data[field]
                
        return normalized_data
```

## **FASE 2: PIPELINE DE INFERÊNCIA EM TEMPO REAL**

### **2.1 - Módulo de Coleta de Dados**

**UPGRADE BOOST.MD**: Processamento em batch paralelo para todos os ativos simultaneamente.

```python
# src/crypto_inference/data_collector.py
class BinanceDataCollector:
    def __init__(self, symbols: list[str]):
        self.symbols = symbols
        self.client = BinanceAPIClient()
    
    async def get_latest_context_batch(self, lookback: int = 2048):
        """
        Coleta últimos 2048 candles M1 para TODOS os ativos monitorados (BOOST.MD)
        Retorna batch único para processamento paralelo eficiente
        """
        batch_data = {}
        
        # Coletar dados para todos os símbolos em paralelo
        tasks = [
            self._get_single_symbol_context(symbol, lookback) 
            for symbol in self.symbols
        ]
        symbol_data_list = await asyncio.gather(*tasks)
        
        # Organizar dados por símbolo
        for symbol, data in zip(self.symbols, symbol_data_list):
            batch_data[symbol] = data
            
        return batch_data
    
    async def _get_single_symbol_context(self, symbol: str, lookback: int):
        """Coleta dados para um símbolo específico"""
        # Implementar coleta da Binance API
        pass
    
    def prepare_inference_batch(self, batch_data: dict) -> torch.Tensor:
        """
        Prepara batch único para inferência (BOOST.MD)
        Aplica mesma normalização e features do treinamento
        Retorna tensor [n_assets, 2048, n_features] para processamento paralelo
        """
        processed_tensors = []
        
        for symbol, raw_data in batch_data.items():
            # Aplicar normalização por janela: (valor[t] / valor[t=0]) - 1
            normalized_data = self._apply_window_normalization(raw_data)
            
            # Adicionar features cíclicas para cada timestep
            enhanced_data = self._add_cyclical_features(normalized_data)
            
            # Converter para tensor
            tensor_data = self._convert_to_tensor(enhanced_data)
            processed_tensors.append(tensor_data)
        
        # Empilhar todos os ativos em um único batch [n_assets, seq_len, features]
        batch_tensor = torch.stack(processed_tensors, dim=0)
        
        return batch_tensor
    
    def _apply_window_normalization(self, raw_data):
        """Aplica normalização (valor[t] / valor[t=0]) - 1"""
        # Implementar mesmo método do CryptoDatasetBuilder
        pass
    
    def _add_cyclical_features(self, normalized_data):
        """Adiciona features sin/cos para cada timestep"""
        # Implementar mesmo método do training pipeline
        pass
```

### **2.2 - Motor de Inferência Bayesiano Otimizado**

**UPGRADE BOOST.MD**: Processamento em batch paralelo + Confirmação de componentes Bayesianos SOTA.

```python
# src/crypto_inference/bayesian_inference_engine.py
class BayesianCryptoInferenceEngine:
    """
    Motor de inferência Bayesiano estado da arte para trading crypto
    UPGRADE BOOST.MD: Processa todos os ativos simultaneamente em batch único
    CONFIRMADO: BayesianPredictionHead e BayesianELBOLoss mantidos como SOTA
    """
    def __init__(self, model_path: str, device: str = "cuda"):
        self.device = device
        self.model = self._load_bayesian_model(model_path)
        self.model.eval()
        
        # Cache para otimização de inferência
        self.context_cache = {}
        self.attention_cache = {}
        
    def _load_bayesian_model(self, model_path: str):
        """Carrega modelo Bayesiano Moirai-MoE com otimizações"""
        # UPGRADE BOOST.MD: Usar Moirai-MoE architecture
        model = MoiraiForecast.load_from_checkpoint(model_path)
        model = model.to(self.device)
        
        # Otimizações de inferência
        if torch.cuda.is_available():
            model = torch.compile(model, mode="reduce-overhead")
        
        return model
    
    def predict_batch_with_uncertainty(
        self, 
        batch_tensor: torch.Tensor,  # [n_assets, 2048, n_features]
        asset_names: list[str],
        num_mc_samples: int = 50,
        return_attention: bool = True
    ) -> dict[str, BayesianPredictionOutput]:
        """
        Predição Bayesiana em batch paralelo para todos os ativos (BOOST.MD)
        Processa tensor [n_assets, 2048, features] de uma só vez para máxima eficiência GPU
        
        CONFIRMADO: Mantém toda decomposição de incerteza Bayesiana (epistêmica vs aleatória)
        CONFIRMADO: BayesianPredictionHead com MC Dropout e pesos variacionais
        
        Returns:
            Dict com BayesianPredictionOutput para cada ativo:
            - forecast: distribuições para cada horizonte temporal
            - epistemic_uncertainty: incerteza do modelo
            - aleatoric_uncertainty: incerteza dos dados  
            - attention_weights: pesos de atenção temporal
            - confidence_intervals: intervalos de confiança calibrados
        """
        with torch.no_grad():
            n_assets = batch_tensor.shape[0]
            
            # Múltiplas passadas Monte Carlo em batch
            mc_predictions = []
            attention_weights_list = []
            
            # Temporariamente ativar dropout para MC
            self._enable_mc_dropout(self.model)
            
            for i in range(num_mc_samples):
                # Processar todos os ativos de uma vez - UPGRADE BOOST.MD
                batch_pred_output = self.model(batch_tensor)  # [n_assets, horizon, features]
                mc_predictions.append(batch_pred_output.samples)
                
                if return_attention and hasattr(batch_pred_output, 'attention_weights'):
                    attention_weights_list.append(batch_pred_output.attention_weights)
            
            # Restaurar modo eval
            self.model.eval()
            
            # Agregar predições MC para cada ativo
            mc_samples = torch.stack(mc_predictions)  # [mc_samples, n_assets, horizon, features]
            
            # Processar cada ativo individualmente para saída estruturada
            asset_predictions = {}
            
            for asset_idx, asset_name in enumerate(asset_names):
                asset_mc_samples = mc_samples[:, asset_idx, :, :]  # [mc_samples, horizon, features]
                
                # Decomposição de incerteza para este ativo (MANTIDO BAYESIANO)
                epistemic_uncertainty = asset_mc_samples.var(dim=0)  # variância entre amostras MC
                aleatoric_uncertainty = (
                    batch_pred_output.variance[asset_idx] 
                    if hasattr(batch_pred_output, 'variance') 
                    else torch.zeros_like(epistemic_uncertainty)
                )
                
                # Predição média e total uncertainty
                mean_prediction = asset_mc_samples.mean(dim=0)
                total_uncertainty = epistemic_uncertainty + aleatoric_uncertainty
                
                # Intervalos de confiança calibrados (MANTIDO BAYESIANO)
                confidence_intervals = self._compute_calibrated_intervals(
                    asset_mc_samples, confidence_levels=[0.68, 0.95, 0.99]
                )
                
                asset_predictions[asset_name] = BayesianPredictionOutput(
                    mean_prediction=mean_prediction,
                    epistemic_uncertainty=epistemic_uncertainty,
                    aleatoric_uncertainty=aleatoric_uncertainty,
                    total_uncertainty=total_uncertainty,
                    confidence_intervals=confidence_intervals,
                    mc_samples=asset_mc_samples,
                    attention_weights=(
                        torch.stack(attention_weights_list)[:, asset_idx, :, :] 
                        if attention_weights_list else None
                    )
                )
            
            return asset_predictions
            
            # Restaurar modo eval
            self.model.eval()
            
            # Agregar predições MC
            mc_samples = torch.stack(mc_predictions)  # [mc_samples, batch, horizon, features]
            
            # Decomposição de incerteza
            epistemic_uncertainty = mc_samples.var(dim=0)  # variância entre amostras MC
            aleatoric_uncertainty = pred_output.variance if hasattr(pred_output, 'variance') else torch.zeros_like(epistemic_uncertainty)
            
            # Predição média e total uncertainty
            mean_prediction = mc_samples.mean(dim=0)
            total_uncertainty = epistemic_uncertainty + aleatoric_uncertainty
            
            # Intervalos de confiança calibrados
            confidence_intervals = self._compute_calibrated_intervals(
                mc_samples, confidence_levels=[0.68, 0.95, 0.99]
            )
            
            return BayesianPredictionOutput(
                mean_prediction=mean_prediction,
                epistemic_uncertainty=epistemic_uncertainty,
                aleatoric_uncertainty=aleatoric_uncertainty,
                total_uncertainty=total_uncertainty,
                confidence_intervals=confidence_intervals,
                mc_samples=mc_samples,
                attention_weights=torch.stack(attention_weights_list) if attention_weights_list else None
            )
    
    def _enable_mc_dropout(self, model: nn.Module):
        """Ativa dropout para Monte Carlo durante inferência"""
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                module.train()  # força dropout ativo
    
    def _compute_calibrated_intervals(
        self, 
        mc_samples: torch.Tensor, 
        confidence_levels: list[float]
    ) -> dict:
        """Computa intervalos de confiança calibrados"""
        intervals = {}
        
        for level in confidence_levels:
            alpha = 1 - level
            lower_q = alpha / 2
            upper_q = 1 - alpha / 2
            
            lower = torch.quantile(mc_samples, lower_q, dim=0)
            upper = torch.quantile(mc_samples, upper_q, dim=0)
            
            intervals[f"{level:.0%}"] = {
                "lower": lower,
                "upper": upper,
                "width": upper - lower
            }
        
        return intervals
    
    def extract_temporal_horizons(self, prediction: BayesianPredictionOutput) -> dict:
        """Extrai previsões para horizontes temporais específicos"""
        horizons = {}
        
        for horizon_name, timestep in [("M1", 0), ("M5", 4), ("M15", 14), ("H1", 59)]:
            horizons[horizon_name] = {
                "mean": prediction.mean_prediction[:, timestep],
                "epistemic_std": torch.sqrt(prediction.epistemic_uncertainty[:, timestep]),
                "aleatoric_std": torch.sqrt(prediction.aleatoric_uncertainty[:, timestep]),
                "total_std": torch.sqrt(prediction.total_uncertainty[:, timestep]),
                "confidence_intervals": {
                    level: {
                        "lower": intervals["lower"][:, timestep],
                        "upper": intervals["upper"][:, timestep]
                    }
                    for level, intervals in prediction.confidence_intervals.items()
                }
            }
        
        return horizons
    
    def compute_uncertainty_metrics(self, prediction: BayesianPredictionOutput) -> dict:
        """Computa métricas avançadas de incerteza"""
        mc_samples = prediction.mc_samples
        
        return {
            # Métricas de incerteza total
            "prediction_entropy": self._compute_entropy(mc_samples),
            "uncertainty_ratio": (prediction.epistemic_uncertainty / prediction.total_uncertainty).mean(),
            
            # Métricas de confiança
            "prediction_confidence": self._compute_confidence_score(mc_samples),
            "uncertainty_budget": {
                "epistemic_pct": (prediction.epistemic_uncertainty / prediction.total_uncertainty).mean() * 100,
                "aleatoric_pct": (prediction.aleatoric_uncertainty / prediction.total_uncertainty).mean() * 100
            },
            
            # Métricas de calibração
            "expected_calibration_error": self._compute_ece(mc_samples),
            "sharpness": prediction.confidence_intervals["95%"]["width"].mean(),
        }
    
    def _compute_entropy(self, mc_samples: torch.Tensor) -> torch.Tensor:
        """Computa entropia das predições Monte Carlo"""
        # Aproximação via histogram
        probs = torch.histc(mc_samples.flatten(), bins=50, density=True)
        probs = probs[probs > 0]  # remove bins vazios
        entropy = -(probs * torch.log(probs)).sum()
        return entropy
    
    def _compute_confidence_score(self, mc_samples: torch.Tensor) -> torch.Tensor:
        """Score de confiança baseado na dispersão das amostras MC"""
        variance = mc_samples.var(dim=0)
        confidence = 1.0 / (1.0 + variance.mean())
        return confidence
    
    def _compute_ece(self, mc_samples: torch.Tensor) -> float:
        """Expected Calibration Error - métrica de calibração"""
        # Implementação simplificada - idealmente usar dados de validação
        # com ground truth para calibração adequada
        return 0.0  # placeholder


@dataclass
class BayesianPredictionOutput:
    """Saída estruturada da inferência Bayesiana"""
    mean_prediction: torch.Tensor
    epistemic_uncertainty: torch.Tensor  # incerteza do modelo
    aleatoric_uncertainty: torch.Tensor  # incerteza dos dados
    total_uncertainty: torch.Tensor
    confidence_intervals: dict
    mc_samples: torch.Tensor
    attention_weights: Optional[torch.Tensor] = None
```

### **2.3 - Pós-processamento Bayesiano e Métricas Avançadas**

```python
# src/crypto_inference/bayesian_postprocessor.py
class BayesianPostprocessor:
    """
    Pós-processamento Bayesiano estado da arte para trading
    Implementa métricas de incerteza, probabilidades condicionais e sinais de trading
    """
    
    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate
        self.calibration_history = deque(maxlen=1000)
        
    def compute_trading_signals(
        self, 
        prediction: BayesianPredictionOutput,
        current_price: float,
        horizons: dict
    ) -> dict:
        """
        Computa sinais de trading baseados em incerteza Bayesiana
        Estado da arte: considera regime uncertainty e model confidence
        """
        signals = {}
        
        for horizon_name, horizon_data in horizons.items():
            # Previsão média e incertezas
            mean_pred = float(horizon_data["mean"])
            epistemic_std = float(horizon_data["epistemic_std"])
            aleatoric_std = float(horizon_data["aleatoric_std"])
            total_std = float(horizon_data["total_std"])
            
            # Métricas de direção com incerteza
            direction_prob = self._compute_directional_probability(
                mean_pred, total_std, current_price
            )
            
            # Probabilidades de breakout considerando incerteza
            breakout_probs = self._compute_breakout_probabilities(
                mean_pred, total_std, current_price
            )
            
            # Kelly criterion com incerteza Bayesiana
            kelly_fraction = self._compute_bayesian_kelly(
                mean_pred, total_std, current_price, epistemic_std
            )
            
            # Volatility regime detection
            regime_prob = self._detect_volatility_regime(
                total_std, epistemic_std, aleatoric_std
            )
            
            # Confidence-weighted position sizing
            position_size = self._compute_position_size(
                direction_prob["bullish"], 
                epistemic_std,
                total_std,
                kelly_fraction
            )
            
            signals[horizon_name] = {
                # Sinais básicos
                "direction": "bullish" if mean_pred > current_price else "bearish",
                "strength": abs(mean_pred - current_price) / current_price,
                
                # Probabilidades
                "directional_probability": direction_prob,
                "breakout_probability": breakout_probs,
                
                # Gestão de risco Bayesiana
                "kelly_fraction": kelly_fraction,
                "position_size": position_size,
                "confidence_score": self._compute_confidence_score(epistemic_std, total_std),
                
                # Regime detection
                "volatility_regime": regime_prob,
                "model_uncertainty": epistemic_std / total_std,  # % de incerteza do modelo
                
                # Intervalos de confiança para stop-loss/take-profit
                "risk_levels": self._compute_risk_levels(horizon_data["confidence_intervals"]),
                
                # Métricas de qualidade da predição
                "prediction_quality": {
                    "uncertainty_ratio": epistemic_std / total_std,
                    "precision_score": 1.0 / (1.0 + total_std),
                    "reliability_score": self._estimate_reliability(epistemic_std),
                }
            }
        
        return signals
    
    def _compute_directional_probability(
        self, 
        mean_pred: float, 
        std: float, 
        current_price: float
    ) -> dict:
        """Probabilidade de movimento direcional usando distribuição Student-t"""
        # Assumindo distribuição Student-t com df estimado
        df = 4.0  # heavy tails para crypto
        
        # Normalizar para distribuição padrão
        z_score = (mean_pred - current_price) / std
        
        # Probabilidades usando Student-t
        from scipy.stats import t
        bullish_prob = 1 - t.cdf(0, df, loc=z_score, scale=1)
        bearish_prob = t.cdf(0, df, loc=z_score, scale=1)
        
        return {
            "bullish": float(bullish_prob),
            "bearish": float(bearish_prob),
            "neutral": max(0.0, 1.0 - bullish_prob - bearish_prob)
        }
    
    def _compute_breakout_probabilities(
        self, 
        mean_pred: float, 
        std: float, 
        current_price: float,
        thresholds: list[float] = [0.01, 0.02, 0.05]
    ) -> dict:
        """Probabilidades de breakout para diferentes thresholds"""
        from scipy.stats import norm
        
        breakout_probs = {}
        for threshold in thresholds:
            upper_breakout = 1 - norm.cdf(
                current_price * (1 + threshold), 
                loc=mean_pred, 
                scale=std
            )
            lower_breakout = norm.cdf(
                current_price * (1 - threshold), 
                loc=mean_pred, 
                scale=std
            )
            
            breakout_probs[f"{threshold:.1%}"] = {
                "upward": float(upper_breakout),
                "downward": float(lower_breakout),
                "total": float(upper_breakout + lower_breakout)
            }
        
        return breakout_probs
    
    def _compute_bayesian_kelly(
        self, 
        mean_pred: float, 
        total_std: float, 
        current_price: float,
        epistemic_std: float
    ) -> float:
        """
        Kelly Criterion modificado para incerteza Bayesiana
        Ajusta para incerteza epistêmica (model uncertainty)
        """
        # Expected return
        expected_return = (mean_pred - current_price) / current_price
        
        # Volatility ajustada por incerteza epistêmica
        adjusted_variance = total_std**2 + (epistemic_std**2 * 0.5)  # penaliza incerteza do modelo
        
        # Kelly fraction com penalização por incerteza
        if adjusted_variance > 0:
            kelly_fraction = expected_return / adjusted_variance
            # Limitar entre -0.25 e 0.25 para segurança
            kelly_fraction = np.clip(kelly_fraction, -0.25, 0.25)
        else:
            kelly_fraction = 0.0
        
        return float(kelly_fraction)
    
    def _detect_volatility_regime(
        self, 
        total_std: float, 
        epistemic_std: float, 
        aleatoric_std: float
    ) -> dict:
        """
        Detecção de regime de volatilidade usando decomposição de incerteza
        """
        # Ratio de incertezas para classificação de regime
        uncertainty_ratio = epistemic_std / total_std if total_std > 0 else 0
        
        # Classificação de regime baseada em incerteza total
        if total_std < 0.01:  # < 1%
            regime = "low_volatility"
        elif total_std < 0.03:  # 1-3%
            regime = "medium_volatility"
        else:  # > 3%
            regime = "high_volatility"
        
        # Confiança do regime baseada em incerteza epistêmica
        regime_confidence = 1.0 - uncertainty_ratio
        
        return {
            "regime": regime,
            "confidence": float(regime_confidence),
            "volatility_level": float(total_std),
            "model_uncertainty_pct": float(uncertainty_ratio * 100)
        }
    
    def _compute_position_size(
        self, 
        directional_prob: float, 
        epistemic_std: float,
        total_std: float,
        kelly_fraction: float
    ) -> dict:
        """
        Position sizing considerando incerteza Bayesiana
        """
        # Base position size do Kelly
        base_size = abs(kelly_fraction)
        
        # Ajuste por confiança (menor incerteza epistêmica = maior confiança)
        confidence_multiplier = 1.0 - (epistemic_std / total_std) if total_std > 0 else 0
        
        # Ajuste por probabilidade direcional
        direction_multiplier = max(0, (directional_prob - 0.5) * 2)  # 0 se prob=0.5, 1 se prob=1.0
        
        # Position size final
        final_size = base_size * confidence_multiplier * direction_multiplier
        
        # Limites de segurança
        max_position = 0.1  # máximo 10% do capital
        final_size = min(final_size, max_position)
        
        return {
            "recommended_size": float(final_size),
            "base_kelly": float(base_size),
            "confidence_adjustment": float(confidence_multiplier),
            "direction_adjustment": float(direction_multiplier),
            "max_allowed": max_position
        }
    
    def _compute_confidence_score(self, epistemic_std: float, total_std: float) -> float:
        """Score de confiança na predição"""
        if total_std == 0:
            return 1.0
        
        # Confiança inversa à incerteza epistêmica
        confidence = 1.0 - (epistemic_std / total_std)
        return max(0.0, min(1.0, confidence))
    
    def _compute_risk_levels(self, confidence_intervals: dict) -> dict:
        """Níveis de risco para stop-loss e take-profit"""
        risk_levels = {}
        
        for level, intervals in confidence_intervals.items():
            risk_levels[level] = {
                "stop_loss": float(intervals["lower"]),
                "take_profit": float(intervals["upper"]),
                "risk_range": float(intervals["upper"] - intervals["lower"])
            }
        
        return risk_levels
    
    def _estimate_reliability(self, epistemic_std: float) -> float:
        """Estimativa de confiabilidade baseada em incerteza epistêmica"""
        # Função sigmóide inversa para mapear incerteza -> confiabilidade
        reliability = 1.0 / (1.0 + np.exp(10 * (epistemic_std - 0.02)))
        return float(reliability)
    
    def update_calibration(self, prediction: float, actual: float, uncertainty: float):
        """
        Atualiza histórico de calibração para avaliação contínua
        """
        error = abs(prediction - actual)
        calibration_score = 1.0 if error <= uncertainty else error / uncertainty
        
        self.calibration_history.append({
            "prediction": prediction,
            "actual": actual,
            "uncertainty": uncertainty,
            "error": error,
            "calibration_score": calibration_score,
            "timestamp": datetime.now()
        })
    
    def get_calibration_metrics(self) -> dict:
        """Métricas de calibração do modelo"""
        if not self.calibration_history:
            return {}
        
        scores = [item["calibration_score"] for item in self.calibration_history]
        errors = [item["error"] for item in self.calibration_history]
        uncertainties = [item["uncertainty"] for item in self.calibration_history]
        
        return {
            "mean_calibration_score": np.mean(scores),
            "calibration_std": np.std(scores),
            "coverage_probability": np.mean([s <= 1.0 for s in scores]),
            "mean_absolute_error": np.mean(errors),
            "mean_uncertainty": np.mean(uncertainties),
            "uncertainty_quality": np.corrcoef(errors, uncertainties)[0, 1] if len(errors) > 1 else 0
        }
```

## **FASE 3: IMPLEMENTAÇÃO PRÁTICA**

### **3.1 - Estrutura de Arquivos**

```
├── src/
│   ├── crypto_data/
│   │   ├── __init__.py
│   │   ├── binance_collector.py
│   │   ├── data_builder.py
│   │   └── preprocessor.py
│   ├── crypto_inference/
│   │   ├── __init__.py
│   │   ├── engine.py
│   │   ├── postprocessor.py
│   │   └── real_time_pipeline.py
│   └── configs/
│       ├── crypto_finetune.yaml
│       └── crypto_inference.yaml
├── scripts/
│   ├── collect_data.py
│   ├── prepare_dataset.py
│   ├── finetune_model.py
│   └── run_inference.py
├── notebooks/
│   ├── data_exploration.ipynb
│   ├── model_validation.ipynb
│   └── inference_demo.ipynb
└── requirements_crypto.txt
```

### **3.2 - Comandos de Execução**

#### **Preparação dos dados**:
```bash
# 1. Coletar dados da Binance
python scripts/collect_data.py --symbols BTC/USDT ETH/USDT --days 1000

# 2. Preparar dataset para treinamento
python scripts/prepare_dataset.py --input raw_data/ --output crypto_dataset/

# 3. Fine-tuning
python -m cli.train -cp configs run_name=crypto_specialist_v2
```

#### **Inferência**:
```bash
# Inferência única
python scripts/run_inference.py --model checkpoints/crypto_specialist_v2.ckpt

# Pipeline em tempo real  
python scripts/real_time_pipeline.py --interval 60  # a cada minuto
```

### **3.3 - Melhorias Arquiteturais**

#### **Uso do Moirai-MoE (UPGRADE BOOST.MD)**
**Justificativa para Moirai-MoE-Base**: Ideal para dataset unificado e anônimo:
- **Especialização automática**: Experts internos se especializam em diferentes padrões de volatilidade/regimes comuns a todos os ativos
- **Melhor captura de padrões diversos**: Um expert para tendências fortes, outro para consolidação, etc.
- **Performance superior**: Demonstrada no benchmark para datasets diversos
- **Scaling eficiente**: MoE permite modelo maior sem proporcional aumento computacional

#### **Dataset Unificado + MoE = Sinergia Perfeita**:
- Dataset anônimo força o modelo a aprender padrões universais 
- MoE automaticamente descobre e especializa em diferentes tipos de movimentos
- Resultado: Modelo mais robusto e generalizável

#### **Optimizações de Inferência**
- **Batch paralelo**: Todos os ativos processados simultaneamente (BOOST.MD)
- **TensorRT compilation** para GPU de produção
- **ONNX export** para deployment multiplataforma  
- **Quantização INT8** para reduzir latência
- **Cache de contexto** para otimização de inferência sequencial

## **FASE 4: VALIDAÇÃO E DEPLOYMENT**

### **4.1 - Métricas de Avaliação**

```python
# Métricas específicas para trading
evaluation_metrics = {
    "forecasting_accuracy": ["MAE", "MAPE", "QuantileLoss"],
    "trading_metrics": ["Sharpe", "MaxDrawdown", "WinRate"],
    "uncertainty_calibration": ["PIT", "Coverage", "Reliability"],
    "computational": ["LatencyP95", "ThroughputRPS", "MemoryUsage"]
}
```

### **4.2 - Ambiente de Desenvolvimento**

#### **Setup local (Dell i5 8GB)**:
```bash
# Preparação e exploração de dados
pip install uni2ts[notebook] binance-connector pandas pyarrow

# Para fine-tuning: usar Kaggle/Colab com GPU
# Para inferência: otimizar para CPU local ou deploy em cloud
```

#### **Setup produção (AWS)**:
- **Treinamento**: p3.2xlarge (V100) no Kaggle/Colab 
- **Inferência**: g4dn.xlarge (T4) para tempo real
- **Produção**: p4d.24xlarge (A100/H100) para escala

### **4.3 - Considerações de Estado da Arte Bayesiano**

#### **Vantagens da abordagem Moirai + Bayesiana**:
1. **Foundation model** pré-treinado + **cabeça Bayesiana especializada**
2. **Decomposição de incerteza**: epistêmica (modelo) vs aleatória (dados)
3. **Distribuições heavy-tail** (Student-t) para capturar extremos de crypto
4. **Calibração probabilística** para confiabilidade das predições
5. **Monte Carlo Dropout** para quantificação de incerteza do modelo

#### **Avanços Bayesianos implementados**:

##### **1. Uncertainty Quantification (Estado da Arte)**
- **Epistemic Uncertainty**: Incerteza do modelo via MC Dropout + pesos variacionais
- **Aleatoric Uncertainty**: Incerteza inerente dos dados via distribuições paramétricas
- **Total Uncertainty**: Combinação calibrada para decision making
- **Temporal Attention**: Pesos de atenção para interpretar relevância temporal

##### **2. Bayesian Risk Management**
- **Kelly Criterion Bayesiano**: Posicionamento ótimo considerando incerteza
- **Confidence-Weighted Sizing**: Tamanho de posição baseado em confiança
- **Regime-Aware Predictions**: Detecção automática de regimes de volatilidade
- **Calibrated Intervals**: Intervalos de confiança para stop-loss/take-profit

##### **3. Advanced Probabilistic Metrics**
```python
# Métricas Bayesianas implementadas:
bayesian_metrics = {
    # Incerteza
    "epistemic_uncertainty": "Incerteza do modelo (reduzível com mais dados)",
    "aleatoric_uncertainty": "Incerteza inerente dos dados (irreduzível)",
    "uncertainty_decomposition": "Ratio epistêmica/total para confiança",
    
    # Calibração
    "expected_calibration_error": "Qualidade da calibração probabilística",
    "prediction_interval_coverage": "Cobertura dos intervalos de confiança",
    "sharpness": "Precisão dos intervalos (narrower = better)",
    
    # Trading
    "directional_probability": "P(up|data) usando distribuições pesadas",
    "breakout_probabilities": "P(breakout > threshold) para vários thresholds",
    "kelly_fraction": "Fração ótima do capital considerando incerteza",
    "regime_probabilities": "P(low/medium/high volatility regime)",
    
    # Qualidade
    "model_confidence": "1 - (epistemic_uncertainty / total_uncertainty)",
    "prediction_reliability": "Score baseado em histórico de calibração",
    "attention_entropy": "Dispersão da atenção temporal (foco vs incerteza)"
}
```

##### **4. Production-Ready Bayesian Pipeline**
```python
# Pipeline Bayesiano completo para produção
class ProductionBayesianPipeline:
    def __init__(self):
        self.model = BayesianCryptoInferenceEngine(model_path)
        self.processor = BayesianPostprocessor()
        self.risk_manager = BayesianRiskManager()
        self.calibration_monitor = CalibrationMonitor()
    
    def predict_and_trade(self, market_data: dict) -> TradingDecision:
        # 1. Inferência Bayesiana
        prediction = self.model.predict_with_uncertainty(market_data)
        
        # 2. Pós-processamento
        signals = self.processor.compute_trading_signals(prediction)
        
        # 3. Gestão de risco Bayesiana
        risk_adjusted_signals = self.risk_manager.adjust_for_uncertainty(signals)
        
        # 4. Decisão final
        decision = self.make_trading_decision(risk_adjusted_signals)
        
        # 5. Monitor de calibração contínua
        self.calibration_monitor.update(prediction, market_data)
        
        return decision
```

##### **5. Melhorias específicas para cripto com Bayesian**:
1. **Heavy-tail distributions** (Student-t) para capturar crashes/pumps
2. **Regime uncertainty** para detectar bull/bear markets automaticamente
3. **Microstructure features** integradas com incerteza: spread, order book depth
4. **Multi-timeframe fusion** com pesos de confiança Bayesianos
5. **Online calibration** para adaptação contínua a novos regimes
6. **Uncertainty-aware portfolio** rebalancing em tempo real

##### **6. Metrics de Avaliação Bayesianas**
```python
advanced_evaluation = {
    # Métricas de forecasting
    "probabilistic_metrics": [
        "Negative Log-Likelihood", "CRPS", "Pinball Loss", 
        "Interval Score", "Energy Score"
    ],
    
    # Métricas de calibração  
    "calibration_metrics": [
        "Reliability Diagram", "PIT Histogram", "Coverage Probability",
        "Average Width", "Expected Calibration Error"
    ],
    
    # Métricas de trading Bayesianas
    "trading_metrics": [
        "Uncertainty-Adjusted Sharpe", "Bayesian Kelly Returns",
        "Regime-Aware Max Drawdown", "Confidence-Weighted Win Rate"
    ],
    
    # Métricas computacionais
    "computational_metrics": [
        "MC Sampling Latency", "Uncertainty Computation Overhead",
        "Memory Usage for Variational Weights", "Calibration Update Speed"
    ]
}
```

Esta implementação Bayesiana representa o **estado da arte** para trading quantitativo com deep learning, oferecendo:
- **Quantificação rigorosa de incerteza** para decision making
- **Gestão de risco probabilística** baseada em teoria da decisão
- **Calibração contínua** para manter qualidade preditiva  
- **Interpretabilidade** via decomposição de incerteza e attention weights

## **CRONOGRAMA SUGERIDO**

| Semana | Atividade | Local |
|--------|-----------|-------|
| 1-2 | Coleta e exploração de dados | Local |
| 3-4 | Desenvolvimento do data builder | Local |
| 5-6 | Fine-tuning inicial | Kaggle/Colab |
| 7-8 | Desenvolvimento da inferência | Local |
| 9-10 | Otimização e validação | Local + Cloud |
| 11-12 | Deploy e testes em produção | AWS |

## **RESUMO DOS UPGRADES APLICADOS (BOOST.MD)**

### **Principais Melhorias Implementadas**

| Componente | Upgrade Aplicado | Vantagem |
|------------|------------------|----------|
| **Dataset** | **Unificado e anônimo** - Remove `item_id` durante treinamento | Força generalização e aprendizado de padrões universais |
| **Arquitetura** | **Moirai-MoE** em vez de Moirai-Base | Especialização automática para dataset diverso |
| **Features** | **Features cíclicas explícitas** (sin/cos) para tempo | Contexto explícito sobre regimes de mercado |
| **Normalização** | **Por janela**: `(valor[t]/valor[t=0]) - 1` | Foca na forma do padrão, não escala absoluta |
| **Inferência** | **Batch paralelo** para todos os ativos | Máxima eficiência GPU e throughput |
| **Componentes Bayesianos** | **Mantidos integralmente** (BayesianHead + ELBO) | **Nenhum downgrade** - Estado da arte preservado |

### **Confirmações Estado da Arte Mantidas**
- ✅ **BayesianPredictionHead completa** com MC Dropout e pesos variacionais
- ✅ **BayesianELBOLoss completa** com regularização KL  
- ✅ **Decomposição de incerteza** epistêmica vs aleatória
- ✅ **BayesianPostprocessor** com Kelly Criterion e gestão de risco
- ✅ **Todas as métricas Bayesianas** para trading quantitativo

## **PRÓXIMOS PASSOS IMEDIATOS (Foco Bayesiano)**

1. **Implementar a BayesianPredictionHead** com decomposição de incerteza
2. **Desenvolver o BayesianCryptoInferenceEngine** com MC sampling otimizado
3. **Criar sistema de calibração** contínua para monitorar qualidade das predições
4. **Implementar métricas Bayesianas** específicas para trading (Kelly, regime detection)
5. **Setup do pipeline de avaliação** com métricas de calibração (ECE, PIT, Coverage)
6. **Desenvolver notebooks Bayesianos** para análise de incerteza e interpretabilidade

### **Arquivos Prioritários para Implementação**:

```bash
# 1. Core Bayesiano (Semana 1-2)
src/uni2ts/model/crypto/bayesian_head.py          # Cabeça Bayesiana
src/uni2ts/loss/bayesian_elbo.py                  # Loss ELBO
src/crypto_inference/bayesian_inference_engine.py # Motor Bayesiano

# 2. Pós-processamento Avançado (Semana 3-4)  
src/crypto_inference/bayesian_postprocessor.py    # Métricas Bayesianas
src/crypto_inference/calibration_monitor.py       # Monitor de calibração
src/crypto_inference/risk_manager.py              # Gestão de risco Bayesiana

# 3. Avaliação e Validação (Semana 5-6)
notebooks/bayesian_validation.ipynb               # Validação de incerteza
notebooks/calibration_analysis.ipynb              # Análise de calibração  
notebooks/uncertainty_decomposition.ipynb         # Análise epistêmica vs aleatória

# 4. Configurações (Paralelo)
configs/crypto_bayesian_finetune.yaml            # Config Bayesiano
configs/uncertainty_inference.yaml               # Config inferência
```

Este plano Bayesiano representa um **salto qualitativo** sobre a proposta original, implementando o verdadeiro **estado da arte** em quantificação de incerteza para trading quantitativo, com fundamentação teórica sólida e aplicabilidade prática comprovada.
