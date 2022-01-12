"""A file that includes all the services and abstractions for the prediction service."""

import json
import pathlib
from abc import ABC, abstractmethod
from os import PathLike
from typing import Optional, List, Union, Any

import numpy as np
import pandas as pd
import pytorch_lightning as pl
from joblib import load
from pytorch_lightning import LightningModule
from transformers import BertTokenizer
from transformers.tokenization_utils_base import EncodedInput, PreTokenizedInput, TextInput, PreTrainedTokenizerBase

from challenge.service.model import BertFtModel
from challenge.service.torch_datasets import PredictionDataModule
from .util import preprocess


class BasePredictionService(ABC):
    """
    A base class of the testing service.

    Any prediction service should expose at least the following methods.
        prepare_data, to transform the dataframe into numeric data for further processing.
        load_model, to load a previously saved local model.
        evaluate_model, to get predictions for unlabeled data.
        decode_predictions, to inverse-transform the predictions to labels.
        make_csv, to create a CSV file with the header "movie_id, genres", according to the specifications.

    """

    def __init__(self) -> None:
        self.config_path: Optional[Union[Union[str, bytes, PathLike], int]] = None  # type: ignore
        self.config: Optional[Any] = None
        self.movie_ids: Optional[pd.Series] = None
        self.features: Optional[Union[TextInput, PreTokenizedInput, EncodedInput]] = None
        self.reload_config()

    def reload_config(self) -> None:
        """
        Reload the configuration file.

        This is needed when loading the model, because the path in the configuration file is modified by the training
        service, which is independent.
        :return: None.
        """
        self.config_path = pathlib.Path(__file__).parent.absolute() / 'service_config.json'
        self.config = json.load(open(self.config_path))

    def prepare_data(self,
                     dataframe: pd.DataFrame) -> None:
        """
        Transform a dataframe for further processing.

        The part of the transformation presented in this abstract class is model-invariant.
        :param dataframe: A pandas dataframe.
        :return: None.
        """
        self.movie_ids = dataframe['movie_id']
        self.features = dataframe['synopsis'].apply(lambda text: preprocess(text)).values

    @abstractmethod
    def load_model(self) -> None:
        """
        Load a model from a checkpoint.

        Assuming a training was done beforehand, the path to the trained model
        should be in the configuration file. However, checkpoint loading differs for each framework.
        :return:
        """
        pass

    @abstractmethod
    def evaluate_model(self) -> None:
        """
        Evaluate a previously loaded model, saving the numeric predictions.

        :return: None
        """
        pass

    @abstractmethod
    def decode_predictions(self) -> None:
        """
        Decode the numeric predictions in order to get the actual labels.

        :return: None
        """
        pass

    @abstractmethod
    def make_csv(self) -> Any:
        """
        Create a CSV representation of the predictions.

        :return: Optionally, CSV representation, as a string.
        """
        pass


class TorchBertPredictionService(BasePredictionService):
    """An implementation of the training service using pytorch lightning and huggingface transformers."""

    def __init__(self) -> None:
        super().__init__()
        self.tokenizer: PreTrainedTokenizerBase = BertTokenizer.from_pretrained(
            self.config['model_name'])  # type: ignore
        self.batch_size: int = self.config['batch_size']  # type: ignore
        self.max_len: int = self.config['max_len']  # type: ignore
        self.data_module: Optional[LightningModule] = None
        self.model: Optional[LightningModule] = None
        self.predictions: Optional[Union[List[float], List[List[float]]]] = None
        self.decoded_predictions: Optional[List[str]] = None

    def prepare_data(self, dataframe: pd.DataFrame) -> None:
        """
        Prepare data for the numeric processing.

        Besides the implementation-invariant transformations defined in the superclass, the data module is created
        and set up.
        :param dataframe: A pandas dataframe containing the prediction data.
        :return: None
        """
        super(TorchBertPredictionService, self).prepare_data(dataframe=dataframe)
        self.data_module = PredictionDataModule(pred_features=self.features,
                                                tokenizer=self.tokenizer,
                                                max_len=self.max_len,
                                                batch_size=self.batch_size)
        self.data_module.setup()

    def load_model(self) -> None:
        """
        Load the BERT fine-tuned from a checkpoint specified by the model_path in the config and set it to evaluation.

        :return: None.
        """
        self.reload_config()
        model_path = self.config['best_model_path']  # type: ignore
        self.model = BertFtModel.load_from_checkpoint(model_path, **self.config)  # type: ignore
        self.model.eval()  # type: ignore

    def evaluate_model(self) -> None:
        """
        Evaluate the previously-loaded model by calling the predict model.

        :return: None.
        """
        trainer = pl.Trainer(gpus=1)
        self.predictions = trainer.predict(model=self.model, datamodule=self.data_module)

    def decode_predictions(self) -> None:
        """
        Decode the numeric predictions to obtain the actual labels.

        Load the previously saved MultiLabelBinarizer to perform the inverse transformation.
        :return: None.
        """
        encoder = load(self.config["label_binarizer_path"])  # type: ignore

        # Todo - Figure a more elegant way to implement the decoding.
        # Stack all the predictions on the first axis, so shape is (n_samples, n_classes)
        predictions = np.vstack([tensor.cpu().detach().numpy() for tensor in self.predictions])  # type: ignore

        # Create a list with tuples of type (id, class_name) for all the encoder classes
        indexed_classes = [(idx, cl) for idx, cl in enumerate(encoder.classes_)]

        # Create a list of sorted pairs of type (id, prob), sorted by the probability for each prediction.
        sorted_predictions = [sorted([(idx, prob) for idx, prob in
                                      enumerate(prediction)], key=lambda x: x[1], reverse=True)  # type: ignore
                              for prediction in predictions]

        # For each sample, take each of the top 5 predictions, and for each, add the corresponding label.
        decoded_predictions = [[indexed_class[1] for prediction in sorted_prediction[:5]
                                for indexed_class in indexed_classes if prediction[0] == indexed_class[0]]
                               for sorted_prediction in sorted_predictions]

        # Transform each prediction in a list of space-separated strings.
        self.decoded_predictions = [" ".join(decoded_prediction) for decoded_prediction in decoded_predictions]

    def make_csv(self) -> Any:
        """
        Create a Dataframe by joining the movie_ids and decoded_predictions Series and encode it as a CSV.

        :return: Optionally, the CSV representation, as a string.
        """
        df = pd.DataFrame({'movie_id': self.movie_ids,
                           'genres': self.decoded_predictions})
        return df.to_csv(index=False, index_label=False)


class PredictionServiceFactory:
    """A factory class for creating prediction service objects."""

    @staticmethod
    def get_service(**kwargs: int) -> BasePredictionService:
        """
        Create an instance of a specific type of training service based on the kwargs specifications.

        :param kwargs: Keyword arguments.
        :return: An instance of a subclass of BertTrainingService.
        """
        service_name = kwargs.get("service_name", "")
        if service_name == "torch_bert_ft":
            return TorchBertPredictionService()
        else:
            raise ValueError(f"Unsupported service type: {service_name}")
