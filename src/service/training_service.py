"""A file that includes all the services and abstractions for the training service."""

import json
import pathlib
from abc import ABC, abstractmethod
from typing import Optional, Any, Union

import pandas as pd
import pytorch_lightning as pl
from joblib import dump
from pytorch_lightning import LightningDataModule, LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.preprocessing import MultiLabelBinarizer
# from sklearn.model_selection import train_test_split
from skmultilearn.model_selection import iterative_train_test_split
from transformers import BertTokenizer
from transformers.tokenization_utils_base import TextInput, PreTokenizedInput, EncodedInput

from challenge.service.model import BertFtModel
from challenge.service.torch_datasets import TrainDataModule
from .util import preprocess


class BaseTrainingService(ABC):
    """
    A base class of the training service.

    Any train service should expose at least the following methods.
        prepare_data, to transform the dataframe into numeric data for further processing.
        create_model, to instantiate the learning model with default / pre-trained parameters.
        train_model, to do the effective training of the model.
    """

    def __init__(self) -> None:
        self.config_path = pathlib.Path(__file__).parent.absolute() / 'service_config.json'
        self.config = json.load(open(self.config_path))
        self.features: Optional[Union[TextInput, PreTokenizedInput, EncodedInput]] = None
        self.labels: Optional[Any] = None
        self.n_classes: Optional[int] = None

    def prepare_data(self,
                     dataframe: pd.DataFrame) -> None:
        """
        Transform a dataframe further processing.

        The part of the transformation presented in this abstract class is model-invariant.
        :param dataframe: A pandas dataframe.
        :return: None.
        """
        # The only relevant predictor variable is the synopsis. Keep only that and the target variable (genres).
        dataframe = dataframe[['synopsis', 'genres']]
        # Drop any duplicates in the dataset to reduce the noise (if they have the same synopsis).
        dataframe.drop_duplicates(subset=['synopsis'], keep='first', inplace=True)
        # Remove HTML tags, URLs, stop words, punctuations, extra whitespaces and then trim.
        self.features = dataframe['synopsis'].apply(lambda text: preprocess(text)).values
        self.features = self.features.reshape((-1, 1))
        # print(self.features.shape)
        # Split the text in the target variable and use a multi-label binarizer to encode its values.
        dataframe.loc[:, 'genres'] = dataframe.genres.apply(lambda e: e.split())
        encoder = MultiLabelBinarizer()
        self.labels = encoder.fit_transform(dataframe['genres'])
        # Dump the encoder in order to do the inverse transform for the test predictions.
        dump(encoder, self.config["label_binarizer_path"])
        self.n_classes = len(encoder.classes_)

    @abstractmethod
    def create_model(self) -> None:
        """
        Create the learning model and instantiate weights.

        :return: None.
        """
        pass

    @abstractmethod
    def train_and_save_model(self) -> None:
        """
        Train the learning model and save it.

        :return: None.
        """
        pass


class TorchBertTrainingService(BaseTrainingService):
    """An implementation of the training service using pytorch lightning and huggingface transformers."""

    def __init__(self) -> None:
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained(self.config['model_name'])
        self.n_classes: Optional[int] = None
        self.data_module: Optional[LightningDataModule] = None
        self.model: Optional[LightningModule] = None
        self.steps_per_epoch: Optional[int] = None

    def prepare_data(self,
                     dataframe: pd.DataFrame) -> None:
        """
        Prepare data for the numeric processing.

        Besides the implementation-invariant transformations defined in the superclass, the data is split into training
        and validation sets and the datamodule is created and set up.
        :param dataframe: A Pandas dataframe containing the training data.
        :return: None
        """
        super(TorchBertTrainingService, self).prepare_data(dataframe=dataframe)
        train_features, train_labels, val_features, val_labels = iterative_train_test_split(self.features,
                                                                                            self.labels,
                                                                                            test_size=0.2)
        train_features = train_features.flatten()
        val_features = val_features.flatten()
        batch_size = self.config['batch_size']
        self.steps_per_epoch = len(train_features) // batch_size
        self.data_module = TrainDataModule(train_features=train_features,
                                           train_labels=train_labels,
                                           val_features=val_features,
                                           val_labels=val_labels,
                                           tokenizer=self.tokenizer,
                                           max_len=self.config['max_len'],
                                           batch_size=batch_size)
        self.data_module.setup()

    def create_model(self) -> None:
        """
        Instantiate the BertFtModel and stores it in the class instance model.

        :return: None.
        """
        self.model = BertFtModel(n_classes=self.n_classes,
                                 steps_per_epoch=self.steps_per_epoch,
                                 **self.config)  # type: ignore

    def train_and_save_model(self) -> None:
        """
        Train and save the model.

        An early stopping callback is provided to stop the model from training when no progress is detected.
        When the model has finished training, save the path to the best model checkpoint in the configuration file.
        :return: None.
        """
        checkpoint_callback = ModelCheckpoint(monitor="val_loss",
                                              filename='Genres-{epoch:02d}-{val_loss:.2f}',
                                              save_top_k=1,
                                              mode='min')
        early_stop_callback = EarlyStopping(monitor="val_loss")
        trainer = pl.Trainer(max_epochs=self.config['n_epochs'],
                             gpus=1,
                             callbacks=[checkpoint_callback, early_stop_callback])
        trainer.fit(self.model, self.data_module)  # type: ignore
        best_model_path = checkpoint_callback.best_model_path
        self.config['best_model_path'] = best_model_path
        json.dump(self.config, open(self.config_path, mode='w'), indent=4)


class TrainingServiceFactory:
    """A factory class for creating training service objects."""

    @staticmethod
    def get_service(**kwargs: int) -> BaseTrainingService:
        """
        Create an instance of a specific type of training service based on the kwargs specifications.

        :param kwargs: Keyword arguments.
        :return: An instance of a subclass of BertTrainingService.
        """
        service_name = kwargs.get("service_name", "")
        if service_name == "torch_bert_ft":
            return TorchBertTrainingService()
        else:
            raise ValueError(f"Unsupported service type: {service_name}")
