# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from typing import Any, Dict


class Run(ABC):
    def __init__(self, parameters: Dict[str, Any]) -> None:
        self.parameters = parameters
        self.init()

    def init(self) -> None:
        print("Init generals...")
        self._init_general()
        print("Done!")
        print("Init Model...")
        self._init_model()
        print("Done!")
        print("Init Encoder...")
        self._init_encoder()
        print("Done!")
        print("Init Datasets...")
        self._init_datasets()
        print("Done!")
        print("Init Evaluation...")
        self._init_evaluation()
        print("Done!")

    def _init_general(self) -> None:
        pass

    def _init_model(self) -> None:
        pass

    def _init_encoder(self) -> None:
        pass

    def _init_datasets(self) -> None:
        pass

    def _init_evaluation(self) -> None:
        pass

    @abstractmethod
    def launch(self) -> Any:
        pass
