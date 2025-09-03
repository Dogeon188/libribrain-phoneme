from torch import nn
from torchmetrics import Accuracy, Precision, Recall
from pytorch_lightning import LightningModule
from torchmetrics import F1Score

from libribrain_experiments.models.meg2vec import Meg2VecModel
# from .utils import modules_from_config, optimizer_from_config, loss_fn_from_config

N_CHANNELS = 306  # Number of input channels (e.g., EEG channels)
N_CLASSES = 39  # Number of output classes for classification (e.g., phonemes)
SEQLEN = 125  # Number of time steps in each input sequence


class BaseClassificationModule(LightningModule):
    """
    To implement a custom classification module, inherit from this BaseClassificationModule, and override the `forward` and `configure_optimizers` methods, and define your module structure and `loss_fn` in `__init__`.
    You should also register your module in `./__init__.py` to make it available for use.
    """
    loss_fn: nn.Module

    def __init__(self):
        super().__init__()
        self.save_hyperparameters()

        self.accuracy = Accuracy(num_classes=N_CLASSES, task="multiclass")
        self.balanced_accuracy = Accuracy(
            num_classes=N_CLASSES, task="multiclass", average="macro")
        self.f1_micro = F1Score(num_classes=N_CLASSES,
                                task="multiclass", average="micro")
        self.f1_macro = F1Score(num_classes=N_CLASSES,
                                task="multiclass", average="macro")
        self.precision_micro = Precision(
            num_classes=N_CLASSES, task="multiclass", average="micro")
        self.precision_macro = Precision(
            num_classes=N_CLASSES, task="multiclass", average="macro")

        self.binary_accuracy = Accuracy(task="binary")
        self.binary_precision = Precision(task="binary")
        self.binary_recall = Recall(task="binary")
        self.binary_f1 = F1Score(task="binary")

    def forward(self, x):
        raise NotImplementedError("Forward method not implemented.")

    def configure_optimizers(self):
        raise NotImplementedError("Optimizer configuration not implemented.")

    def training_step(self, batch, batch_idx):
        x, y = batch[0], batch[1]
        y_hat = self(x)

        loss = self.loss_fn(y_hat, y)
        acc = self.accuracy(y_hat, y)
        f1_micro = self.f1_micro(y_hat, y)
        f1_macro = self.f1_macro(y_hat, y)
        bal_acc = self.balanced_accuracy(y_hat, y)

        self.log('train_loss', loss)
        self.log('train_acc', acc)
        self.log('train_f1_micro', f1_micro)
        self.log('train_f1_macro', f1_macro)
        self.log('train_precision_micro', self.precision_micro(y_hat, y))
        self.log('train_precision_macro', self.precision_macro(y_hat, y))
        self.log('train_bal_acc', bal_acc)

        for class_idx in range(y_hat.shape[1]):
            y_binary = (y == class_idx).int()
            y_hat_binary = y_hat.argmax(dim=1) == class_idx
            binary_acc = self.binary_accuracy(y_hat_binary, y_binary)
            self.log(f'train_acc_class_{class_idx}', binary_acc)

            monitored_class_prediction_rate = y_hat_binary.float().mean()
            self.log(f'train_prediction_rate_class_{class_idx}',
                     monitored_class_prediction_rate)
            monitored_class_recall = self.binary_recall(y_hat_binary, y_binary)
            self.log(f'train_recall_class_{class_idx}', monitored_class_recall)
            monitored_class_precision = self.binary_precision(
                y_hat_binary, y_binary)
            self.log(f'train_precision_class_{class_idx}',
                     monitored_class_precision)
            y_hat_probs = nn.functional.softmax(y_hat, dim=1)
            monitored_class_mean_probs = y_hat_probs[:, class_idx].mean(
            )
            self.log(f'train_mean_probability_class_{class_idx}',
                     monitored_class_mean_probs)
            self.log(f'train_f1_class_{class_idx}', self.binary_f1(
                y_hat_binary, y_binary))
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch[0], batch[1]
        y_hat = self(x)

        loss = self.loss_fn(y_hat, y)
        acc = self.accuracy(y_hat, y)
        f1_micro = self.f1_micro(y_hat, y)
        f1_macro = self.f1_macro(y_hat, y)
        bal_acc = self.balanced_accuracy(y_hat, y)

        self.log('val_loss', loss)
        self.log('val_acc', acc)
        self.log('val_f1_micro', f1_micro)
        self.log('val_f1_macro', f1_macro)
        self.log('val_precision_micro', self.precision_micro(y_hat, y))
        self.log('val_precision_macro', self.precision_macro(y_hat, y))
        self.log('val_bal_acc', bal_acc)

        for class_idx in range(y_hat.shape[1]):
            y_binary = (y == class_idx).int()
            y_hat_binary = y_hat.argmax(dim=1) == class_idx
            binary_acc = self.binary_accuracy(y_hat_binary, y_binary)
            self.log(f'val_acc_class_{class_idx}', binary_acc)
            monitored_class_prediction_rate = y_hat_binary.float().mean()
            self.log(f'val_prediction_rate_class_{class_idx}',
                     monitored_class_prediction_rate)
            monitored_class_recall = self.binary_recall(y_hat_binary, y_binary)
            self.log(f'val_recall_class_{class_idx}', monitored_class_recall)
            monitored_class_precision = self.binary_precision(
                y_hat_binary, y_binary)
            self.log(f'val_precision_class_{class_idx}',
                     monitored_class_precision)
            self.log(f'val_f1_class_{class_idx}',
                     self.binary_f1(y_hat_binary, y_binary))
            y_hat_probs = nn.functional.softmax(y_hat, dim=1)
            monitored_class_mean_probs = y_hat_probs[:, class_idx].mean(
            )
            self.log(f'val_mean_probability_class_{class_idx}',
                     monitored_class_mean_probs)

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        loss = self.loss_fn(y_hat, y)
        acc = self.accuracy(y_hat, y)
        f1_micro = self.f1_micro(y_hat, y)
        f1_macro = self.f1_macro(y_hat, y)
        bal_acc = self.balanced_accuracy(y_hat, y)

        self.log('test_loss', loss)
        self.log('test_acc', acc)
        self.log('test_f1_micro', f1_micro)
        self.log('test_f1_macro', f1_macro)
        self.log('test_precision_micro', self.precision_micro(y_hat, y))
        self.log('test_precision_macro', self.precision_macro(y_hat, y))
        self.log('test_bal_acc', bal_acc)

        for class_idx in range(y_hat.shape[1]):
            y_binary = (y == class_idx).int()
            y_hat_binary = y_hat.argmax(dim=1) == class_idx
            binary_acc = self.binary_accuracy(y_hat_binary, y_binary)
            self.log(f'test_acc_class_{class_idx}', binary_acc)
            monitored_class_prediction_rate = y_hat_binary.float().mean()
            self.log(f'test_prediction_rate_class_{class_idx}',
                     monitored_class_prediction_rate)
            monitored_class_recall = self.binary_recall(y_hat_binary, y_binary)
            self.log(f'test_recall_class_{class_idx}', monitored_class_recall)
            monitored_class_precision = self.binary_precision(
                y_hat_binary, y_binary)
            self.log(f'test_precision_class_{class_idx}',
                     monitored_class_precision)
            y_hat_probs = nn.functional.softmax(y_hat, dim=1)
            monitored_class_mean_probs = y_hat_probs[:, class_idx].mean(
            )
            self.log(f'test_mean_probability_class_{class_idx}',
                     monitored_class_mean_probs)

            self.log(f'test_f1_class_{class_idx}',
                     self.binary_f1(y_hat_binary, y_binary))
        return loss
