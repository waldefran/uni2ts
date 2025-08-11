Resumo de viabilidade e aderência SOTA
- A arquitetura proposta (Moirai-MoE + cabeça Bayesiana + ELBO) alinha com o padrão SOTA divulgado na documentação e nos planos: `uni2ts.model.crypto.bayesian_head.BayesianPredictionHead`, `uni2ts.loss.bayesian_elbo.BayesianELBOLoss`, `uni2ts.callbacks.bayesian_uncertainty.BayesianUncertaintyMonitor`, `uni2ts.data.builder.crypto.CryptoDatasetBuilder`.
- O notebook cobre o pipeline end-to-end, mas há pontos de fragilidade que podem quebrar a execução e/ou reduzir a reprodutibilidade e o rigor Bayesiano. Alguns trechos estão “otimizados para demonstração” e não para produção.
- Conclusão: a direção é SOTA, mas não está pronta para “SOTA de produção” sem os ajustes abaixo (consistência de schema de batch, integração Lightning, checagem de métricas/monitor, salvamento/restore, e precisão mista + dataloader tuning).

Pontos fortes
- Dataset unificado + anonimização + normalização por janela + features cíclicas configurados no notebook em linha com a doc SOTA: ver CRYPTO_README.md e IMPLEMENTATION_STATUS.md.
- Callbacks Bayesianos e annealing já integrados: `uni2ts.callbacks.bayesian_uncertainty.BayesianUncertaintyMonitor` e `uni2ts.callbacks.elbo_annealing.ELBOAnnealingCallback`.
- Testes utilitários de pipeline presentes: test_sota_pipeline.py e test_simple.py.
- Downloader dedicado e estrutura de dados coerente: binanceDataloader.py.

Riscos e correções prioritárias
1) Inconsistência de schema de batch (x/y vs past_target/future_target)
- Problema: no EDA você acessa chaves ‘past_target/future_target’, mas na criação do modelo assume ‘x/y’. Depois cria um wrapper para converter. Isso pode quebrar dependendo do que `uni2ts.data.builder.crypto.CryptoDatasetBuilder` retorna.
- Ação: padronizar o pipeline para um único schema (recomendado: past_target/past_observed_target/future_target) e remover o wrapper no fit/test/visualização.

2) Fallback para um “dummy” MoiraiMoEModule
- Problema: se o import falhar, uma classe simplificada é criada. Isso quebra a premissa SOTA (não haverá cabeça Bayesiana, atenção temporal, etc.).
- Ação: falhar explicitamente com erro orientado (fail-fast) e garantir que o módulo real existe. Não faça fallback para dummy.

3) Duplicidade/risco na integração Lightning
- Problema: você instancia um modelo “puro” + otimizador + loss fora do Lightning, mas treina com `MoiraiBayesianLightningModule` (importado via scripts). O otimizador e a loss criados fora ficam mortos, e os callbacks “val/ELBO” dependem do nome de métrica logado dentro do LightningModule.
- Ação: centralizar toda configuração de loss/otimizador/monitor dentro do LightningModule. Assegure que o módulo logue exatamente as chaves monitoradas em checkpoint/early stopping (ex.: “val/ELBO” vs “val/loss”).

4) Monitor/metric names
- Problema: “val/ELBO” é usado nos callbacks, mas é comum LightningModule logar “val/loss”. Isso pode impedir checkpoint/early stopping.
- Ação: alinhar o nome da métrica no LightningModule com os callbacks ou ajustar os callbacks no notebook para a chave correta.

5) Salvamento/restauração de modelo
- Problema: salvar state_dict com torch.save no LightningModule e em um “modelo base” desconectado é frágil. O padrão Lightning é salvar .ckpt com trainer.save_checkpoint.
- Ação: usar Trainer.save_checkpoint e garantir que o carregamento posterior use o mesmo LightningModule + hparams.

6) Leitura de métricas
- Problema: caminho fixo para “version_0/metrics.csv”. Em execuções repetidas, a versão muda.
- Ação: glob versão mais recente.

7) Visualização com StudentT
- Problema: a função assume que pl_module(model_batch) retorna um objeto com loc/scale/df. Isso depende da assinatura real do LightningModule. Pode ser que exista um predict_step/forward que retorne algo diferente (amostras MC, quantis, etc.).
- Ação: padronizar a API de inferência (ideal: predict_step retornar distribuição ou quantis). Se usar Student-T, usar quantis via df real; se a cabeça já devolve quantis, usar diretamente.

8) Performance de DataLoader
- Ação: ativar persistent_workers, prefetch_factor e precision mista (“16-mixed”), especialmente no Kaggle.

9) Reprodutibilidade
- Ação: setar seed com pl.seed_everything(workers=True), registrar versões e carimbos de tempo no log.

10) Hardcoding/Config
- Ação: evitar paths e listas (assets, plot_dir) hardcoded no código e mover para o YAML, conforme já sugerido em code_review.md.

Patches recomendados (mínimos e seguros)

1) Fail-fast no import do modelo (evitar dummy)

````python
# ...existing code...
try:
    from uni2ts.model.moirai_moe import MoiraiMoEModule
except Exception as e:
    raise ImportError(f"Importação do MoiraiMoEModule falhou: {e}. Verifique a instalação do uni2ts.") from e
# ...existing code...
````

2) Precision mista e tuning dos DataLoaders

````python
# ...existing code...
trainer_config["precision"] = "16-mixed" if torch.cuda.is_available() else 32
# ...existing code...
prefetch_factor = 2 if num_workers > 0 else None
persistent_workers = num_workers > 0

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=persistent_workers,
    prefetch_factor=prefetch_factor,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=persistent_workers,
    prefetch_factor=prefetch_factor,
)
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=persistent_workers,
    prefetch_factor=prefetch_factor,
)
# ...existing code...
````

3) Seed e determinismo

````python
# ...existing code...
import random
import numpy as np
pl.seed_everything(42, workers=True)
torch.backends.cudnn.benchmark = True
# ...existing code...
````

4) Checagem de métricas (version_* dinâmico)

````python
# ...existing code...
import glob
metrics_base = os.path.join(LOG_DIR, "crypto_finetune")
versions = sorted(glob.glob(os.path.join(metrics_base, "version_*")), key=os.path.getmtime)
metrics_path = os.path.join(versions[-1], "metrics.csv") if versions else None

if metrics_path and os.path.exists(metrics_path):
    metrics_df = pd.read_csv(metrics_path)
    # ...existing code...
else:
    print(f"Nenhum metrics.csv encontrado em {metrics_base}")
# ...existing code...
````

5) Salvamento correto de checkpoint

````python
# ...existing code...
ckpt_dir = os.path.join(OUTPUT_DIR, "checkpoints")
os.makedirs(ckpt_dir, exist_ok=True)
ckpt_path = os.path.join(ckpt_dir, "crypto_moirai_moe_bayesian.ckpt")
trainer.save_checkpoint(ckpt_path)
print(f"Checkpoint salvo em {ckpt_path}")
# ...existing code...
````

6) Tornar plot_dir configurável e usar o logger do trainer no callback

````python
# ...existing code...
class BayesianUncertaintyMonitor(Callback):
    def __init__(self, ..., plot_dir: str = "./uncertainty_plots", **kwargs):
        super().__init__()
        self.plot_dir = plot_dir
        os.makedirs(self.plot_dir, exist_ok=True)
# ...existing code...
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        # ...existing code...
        if trainer.logger:
            trainer.logger.log_metrics(epoch_metrics, step=trainer.global_step)
# ...existing code...
````

Checagens rápidas que recomendo executar
- Verificar se o LightningModule realmente loga “val/ELBO” (ou ajustar callbacks): finetune_model.py.
- Rodar os testes utilitários em sequência:
  - python test_simple.py
  - python test_sota_pipeline.py
- Confirmar o schema de batch do `uni2ts.data.builder.crypto.CryptoDatasetBuilder` e remover o BatchConverterDataLoader quando o schema já for o esperado.
- Garantir que a visualização use a mesma API do forward/predict do LightningModule (se retorna distribuição ou quantis); se for Student-T, use df do modelo (você já usa `uni2ts.distribution.student_t` na visualização, o que é bom).
