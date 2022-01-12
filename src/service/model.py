"""A file that includes all the models to be used in the training/inference services."""
from typing import Optional, Any, Tuple, List

import pytorch_lightning as pl
import torch.nn as nn
from pytorch_lightning.utilities.types import STEP_OUTPUT
from transformers import BertModel, AdamW, get_linear_schedule_with_warmup


class BertFtModel(pl.LightningModule):
    """A fine-tuned BERT model for multi-label classification."""

    def __init__(self,
                 n_classes: int,
                 steps_per_epoch: int,
                 *args: str,
                 **kwargs: int):
        """
        Initializes a BertFtModel object.

        :param n_classes: The number of classes - Output size of the classification layer.
        :param steps_per_epoch: The number of steps in each epoch (effectively n_samples/batch_size)
        :param args: Non-keyword varargs.
        :param kwargs: Keyword varargs.
        """
        super(BertFtModel, self).__init__()
        self.save_hyperparameters('n_classes', 'steps_per_epoch')

        model_name = kwargs.get("model_name")

        # Initialize the feature extraction part of the model, based on BERT
        self.featurizer = BertModel.from_pretrained(model_name, return_dict=True)
        self.dropout = nn.Dropout(p=0.1)  # type: ignore

        # Initialize the classifier, which is a linear layer.
        # For multi-label classification, use sigmoid as activation instead of softmax
        self.classifier = nn.Linear(self.featurizer.config.hidden_size, n_classes)  # type: ignore
        self.n_epochs = kwargs.get("n_epochs")
        self.learning_rate = kwargs.get("learning_rate")
        self.steps_per_epoch = steps_per_epoch

        # BCEWithLogitsLoss combines a sigmoid layer with a BCELoss.
        self.criterion = nn.BCEWithLogitsLoss()  # type: ignore

    def forward(self,
                input_ids: Any,
                attn_mask: Any) -> Any:  # type: ignore
        """
        Forward-propagates the current batch tensor through the architecture.

        :param input_ids: A tensor containing the indices of the tokens in the batch.
        :param attn_mask: A binary mask that indicates to which tokens the model should attend.
        :return: The predictions of the last FC layer (un-normalized).
        """
        features = self.featurizer(input_ids=input_ids, attention_mask=attn_mask)
        # When classifying, only take the [CLS] (pooler) token and take it through the linear layer
        return self.dropout(self.classifier(features.pooler_output))
        # return self.classifier(features.pooler_output)

    def training_step(self,
                      batch: Any,
                      batch_idx: int) -> STEP_OUTPUT:  # type: ignore
        """
        Train the model for one step..

        :param batch_idx: Unused.
        :param batch: A dictionary corresponding to the current batch (1 or more items in the dataset), containing
        the id s of the tokens in the input sequence, attention mask and optionally labels.
        :return: A dictionary containing the loss, the predictions and the ground truth labels.
        """
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['label']
        outputs = self(input_ids, attention_mask)
        loss = self.criterion(outputs, labels)
        self.log('train_loss', loss, prog_bar=True, logger=True)

        return {"loss": loss, "predictions": outputs, "labels": labels}

    def validation_step(self,
                        batch: Any,
                        batch_idx: int) -> STEP_OUTPUT:  # type: ignore
        """
        Validate the model for one step.

        :param batch_idx: Unused.
        :param batch: A dictionary corresponding to the current batch (1 or more items in the dataset), containing
        the id s of the tokens in the input sequence, attention mask and optionally labels.
        :return: The loss value.
        """
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['label']
        outputs = self(input_ids, attention_mask)
        loss = self.criterion(outputs, labels)
        self.log('val_loss', loss, prog_bar=True, logger=True)
        return {"loss": loss}

    def predict_step(self,
                     batch: Any,
                     batch_idx: int,
                     dataloader_idx: Optional[int] = None) -> STEP_OUTPUT:
        """
        Predict the labels for one step.

        :param batch: A dictionary corresponding to the current batch (1 or more items in the dataset), containing
        the id s of the tokens in the input sequence, attention mask and optionally labels.
        :param batch_idx: Unused.
        :param dataloader_idx: Unused.
        :return:
        """
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        return self(input_ids, attention_mask)  # type: ignore

    def configure_optimizers(self) -> Tuple[List[Any], List[Any]]:
        """
        Configure the optimizer and scheduler for training.

        :return: A tuple consisting of a list of optimizers and a list of schedulers.
        """
        optimizer = AdamW(self.parameters(), lr=self.learning_rate, eps=1e-8)
        warmup_steps = self.steps_per_epoch // 3
        total_steps = self.steps_per_epoch * self.n_epochs - warmup_steps  # type: ignore
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
        return [optimizer], [scheduler]
