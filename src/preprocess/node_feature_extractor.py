import torch
from safetensors.torch import save_file
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from src.utils import ParameterKeys
import os
import src.data.feature_extraction as data_pkg
import src.models.feature_extraction as model_pkg
from tqdm import tqdm
from src.data.utils import BatchKeys


class NodeFeatureExtractor:
    def __init__(self, parameters: dict):
        self.parameters = parameters
        self.init()

    def init(self) -> None:
        print("Init general...")
        self._init_general()
        print("Done!")
        print("Init dataloader...")
        self._init_dataloader()
        print("Done!")
        print("Init model...")
        self._init_model()
        print("Done!")
        print("Init tokenizer...")
        self._init_tokenizer()
        print("Done!")

    def _init_general(self) -> None:
        general_parameters = self.parameters.get(ParameterKeys.GENERAL)
        self.out_dir = general_parameters.get(ParameterKeys.OUT_DIR, "./")
        os.makedirs(self.out_dir, exist_ok=True)
        self.pbar = general_parameters.get(ParameterKeys.PBAR, False)
        self.node_t = general_parameters.get(ParameterKeys.NODE_INFO)
        self.device = general_parameters.get(ParameterKeys.DEVICE, "cpu")

    def _init_dataloader(self) -> None:
        dataset_parameters = self.parameters.get(ParameterKeys.DATA)
        dataset_name = dataset_parameters.get(ParameterKeys.NAME)
        dataset_config = dataset_parameters.get(ParameterKeys.CFG, dict())
        dataset = data_pkg.__dict__[dataset_name](**dataset_config)
        loader_parameters = self.parameters.get(ParameterKeys.LOADER, dict())
        self.dataloader = DataLoader(
            dataset=dataset,
            collate_fn=getattr(dataset, "collate_fn", None),
            shuffle=False,
            **loader_parameters,
        )

    def _init_model(self) -> None:
        model_parameters = self.parameters.get(ParameterKeys.MODEL)
        model_name = model_parameters.get(ParameterKeys.NAME)
        model_config = model_parameters.get(ParameterKeys.CFG, dict())
        self.model = model_pkg.__dict__[model_name](**model_config).to(self.device)

    def _init_tokenizer(self) -> None:
        tokenizer_parammeters = self.parameters.get(ParameterKeys.TOKENIZER)
        tokenizer_config = tokenizer_parammeters.get(ParameterKeys.CFG, dict())
        self.tokenizer = AutoTokenizer.from_pretrained(**tokenizer_config)
        self.tokenizer_kwargs = tokenizer_parammeters.get(ParameterKeys.KWARGS, dict())

    def get_bar(self):
        if not self.pbar:
            return self.dataloader
        return tqdm(
            self.dataloader,
            total=len(self.dataloader),
            desc=f"Extracting raw node features for node type {self.node_t}",
        )

    def tokenize(self, text: list[str]) -> dict[str, torch.Tensor]:
        return self.tokenizer(text, **self.tokenizer_kwargs).to(self.device)

    def __call__(self) -> None:
        self.launch()

    @torch.no_grad()
    def launch(self) -> None:
        print("Starting feature extraction!")
        bar = self.get_bar()
        features_list = []
        for batch in bar:
            batch_node_features = self.model(self.tokenize(batch)).cpu()
            features_list.append(batch_node_features)
        node_features = torch.cat(features_list, dim=0) if features_list else torch.empty(0)
        save_file(
            {BatchKeys.FEATURES: node_features},
            f"{self.out_dir}/{self.node_t}.safetensors",
        )
        print(f"Saved {node_features.shape[0]} node features to {self.out_dir}/{self.node_t}.safetensors")
