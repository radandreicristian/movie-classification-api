"""A file that includes all the classes needed for PytorchLightning based datasets."""

from typing import Optional, Union, Any, Dict

import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import Dataset, DataLoader
from transformers.tokenization_utils_base import TextInput, PreTokenizedInput, EncodedInput, PreTrainedTokenizerBase


class GenresDataset(Dataset):  # type: ignore
    """A Dataset wrapper for the movie genres data."""

    def __init__(self,
                 text: Union[TextInput, PreTokenizedInput, EncodedInput],
                 tokenizer: PreTrainedTokenizerBase,
                 max_length: int,
                 labels: Optional[torch.Tensor] = None) -> None:
        self.tokenizer = tokenizer
        self.text = text
        self.labels = labels
        self.max_length = max_length

    def __len__(self) -> int:
        """
        Compute the length of the dataset (number of samples).

        :return: The length of the dataset.
        """
        return len(self.text)

    def __getitem__(self,
                    item_idx: int) -> Dict[str, Any]:
        """
        Return an item for the specified index.

        Retrieves and transforms the text at item_idx, returning a dictionary which contains input ids, the attention
        masks. In case labels are provided, also return a tensor containing the labels.
        :param item_idx:
        :return:
        """
        text = self.text[item_idx]
        inputs = self.tokenizer.encode_plus(text=text,
                                            text_pair=None,
                                            add_special_tokens=True,
                                            max_length=self.max_length,
                                            padding='max_length',
                                            return_token_type_ids=False,
                                            return_attention_mask=True,
                                            truncation=True,
                                            return_tensors='pt'
                                            )
        input_ids = inputs['input_ids'].flatten()
        attn_mask = inputs['attention_mask'].flatten()

        if self.labels is None:
            return {'input_ids': input_ids,
                    'attention_mask': attn_mask}
        else:
            return {'input_ids': input_ids,
                    'attention_mask': attn_mask,
                    'label': torch.tensor(self.labels[item_idx], dtype=torch.float)}  # type: ignore


class TrainDataModule(pl.LightningDataModule):
    """A LightningDataModule for the training (labeled) data."""

    def __init__(self,
                 train_features: Union[TextInput, PreTokenizedInput, EncodedInput],
                 train_labels: Any,
                 val_features: Union[TextInput, PreTokenizedInput, EncodedInput],
                 val_labels: Any,
                 tokenizer: PreTrainedTokenizerBase,
                 max_len: int,
                 batch_size: int) -> None:
        super(TrainDataModule, self).__init__()  # type: ignore
        self.train_features, self.train_labels = train_features, train_labels
        self.val_features, self.val_labels = val_features, val_labels
        self.tokenizer = tokenizer
        self.max_len: int = max_len
        self.batch_size: int = batch_size
        self.train_dataset: Optional[Dataset] = None  # type: ignore
        self.val_dataset: Optional[Dataset] = None  # type: ignore

    def setup(self,
              stage: Optional[str] = None) -> None:
        """
        Perform data-related operations (distributed).

        Build the train and validation dataset object.
        :param stage: The stage of learning (fit, test, etc.)
        :return: None.
        """
        self.train_dataset = GenresDataset(text=self.train_features,
                                           labels=self.train_labels,
                                           tokenizer=self.tokenizer,
                                           max_length=self.max_len)
        self.val_dataset = GenresDataset(text=self.val_features,
                                         labels=self.val_labels,
                                         tokenizer=self.tokenizer,
                                         max_length=self.max_len)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        """
        Create a DataLoader for the validation dataset.

        :return: The created dataloader.
        """
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=8)  # type: ignore

    def val_dataloader(self) -> EVAL_DATALOADERS:
        """
        Create a DataLoader for the validation dataset.

        :return: The created dataloader.
        """
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          num_workers=8)  # type: ignore


class PredictionDataModule(pl.LightningDataModule):
    """A LightningDataModule for the prediction (unlabeled) data."""

    def __init__(self, pred_features: Union[TextInput, PreTokenizedInput, EncodedInput],
                 tokenizer: PreTrainedTokenizerBase,
                 max_len: int,
                 batch_size: int):
        super(PredictionDataModule, self).__init__()  # type: ignore
        self.pred_features = pred_features
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.batch_size = batch_size
        self.test_dataset: Optional[Dataset] = None  # type: ignore

    def setup(self,
              stage: Optional[str] = None) -> None:
        """
        Perform data-related operations (distributed).

        Build the test dataset object.
        :param stage: The stage of learning (fit, test, etc.)
        :return: None
        """
        self.test_dataset = GenresDataset(text=self.pred_features,
                                          labels=None,
                                          tokenizer=self.tokenizer,
                                          max_length=self.max_len)

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        """
        Create a DataLoader for the prediction dataset.

        :return: The created dataloader.
        """
        return DataLoader(self.test_dataset,
                          batch_size=self.batch_size,
                          num_workers=8)  # type: ignore
