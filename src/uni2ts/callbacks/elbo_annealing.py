"""
ELBO Annealing Callback SOTA
Automatiza o step_epoch() da BayesianELBOLoss para desacoplar do LightningModule

CORREÇÃO: Remove acoplamento manual entre loss e módulo principal
"""

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from uni2ts.loss.bayesian_elbo import BayesianELBOLoss


class ELBOAnnealingCallback(Callback):
    """
    Callback para automatizar annealing do KL na BayesianELBOLoss
    Remove necessidade de chamar step_epoch() manualmente
    """
    
    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Chama step_epoch() na loss function se ela suportar annealing"""
        
        # Verificar se o módulo tem criterion com step_epoch
        if hasattr(pl_module, 'criterion') and hasattr(pl_module.criterion, 'step_epoch'):
            if isinstance(pl_module.criterion, BayesianELBOLoss):
                pl_module.criterion.step_epoch()
