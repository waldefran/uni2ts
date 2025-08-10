## Pontos de Melhoria e Correções

Apesar da alta qualidade, alguns pontos podem ser refinados para tornar a implementação ainda mais robusta e flexível.

### 1. **Configuração e Hardcoding**
- **`BayesianUncertaintyMonitor`**:
    - **Sugestão:** O diretório de plots (`plot_dir`) está hardcoded como `"./uncertainty_plots"`. O ideal é que este caminho seja configurável via Hydra, assim como outros parâmetros do callback.
    - **Exemplo:**
      ```python
      # Em bayesian_uncertainty.py
      def __init__(self, ..., plot_dir: str = "./uncertainty_plots"):
          ...
      
      # Em finetune_bayesian_moe.yaml
      callbacks:
        - _target_: uni2ts.callbacks.BayesianUncertaintyMonitor
          plot_dir: ${hydra:runtime.output_dir}/uncertainty_plots 
      ```

- **`CryptoDatasetBuilder`**:
    - **Sugestão:** A lista de ativos (`target_assets`) na `CryptoConfig` é fixa. Para maior flexibilidade, ela deveria ser passada como um parâmetro no arquivo de configuração `.yaml`.
    - **Exemplo:**
      ```python
      # Em crypto.py
      @dataclass
      class CryptoConfig:
          ...
          target_assets: List[str] = field(default_factory=lambda: ["BTCUSDT", "ETHUSDT"])
      
      # Em finetune_bayesian_moe.yaml
      data:
        _target_: uni2ts.data.builder.CryptoDatasetBuilder
        config:
          target_assets: ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
      ```

### 2. **Boas Práticas e Robustez**
- **`BayesianUncertaintyMonitor`**:
    - **Problema:** O logging é feito usando `pl.trainer.logger.log_metrics(...)`. O objeto `pl.trainer` é um proxy global que pode ser problemático.
    - **Correção:** O método `on_validation_batch_end` já recebe o objeto `trainer` como argumento. Use `trainer.logger.log_metrics(...)` para garantir que está usando a instância correta do logger associada ao treinamento atual.
    - **Problema:** A linha `matplotlib.use('Agg')` define o backend de plotagem globalmente, o que pode causar efeitos colaterais inesperados em outras partes do código que possam usar `matplotlib`.
    - **Sugestão:** Gerencie o ciclo de vida da figura e do plot localmente dentro da função `_create_uncertainty_plots` para evitar alterações de estado globais.

- **`bayesian_elbo.py`**:
    - **Problema:** O método `step_epoch()` precisa ser chamado manualmente pelo `LightningModule` principal. Isso é um acoplamento implícito e frágil.
    - **Sugestão:** Em vez de um método manual, crie um `Callback` do PyTorch Lightning que se encarregue de chamar `loss_func.step_epoch()` no evento `on_train_epoch_end`. Isso desacopla a loss do ciclo de vida do `LightningModule`.

- **`crypto_bayesian.py`**:
    - **Sugestão:** Os avisos de configuração (`⚠️ Aviso: ...`) são impressos diretamente no `stdout`. Seria mais idiomático usar o módulo `logging` do Python (`logging.warning(...)`). Isso permite que os usuários controlem a verbosidade e redirecionem os logs de forma padronizada.

### 3. **Precisão e Detalhes Técnicos**
- **`bayesian_head.py`**:
    - **Ponto de atenção:** No método `_process_mc_outputs`, o cálculo do intervalo de confiança usa z-scores fixos (ex: 1.96 para 95%). Embora seja uma aproximação comum, a distribuição subjacente é uma Student-T.
    - **Sugestão:** Para maior precisão, use a função `scipy.stats.t.ppf` (percent point function) para obter os quantis corretos da distribuição T, usando os graus de liberdade (`df`) calculados. Isso resultará em intervalos de confiança mais precisos, especialmente para `df` baixos.

---