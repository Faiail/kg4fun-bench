from .run import Run
import src.data as data_pkg
import src.models as model_pkg
from src.scheduler import scheduler_registry
import src.losses as loss_pkg
import src.early_stop as early_stop_pkg
import src.metrics as metrics_pkg
from torchmetrics import MetricCollection
import torch.optim.lr_scheduler as schedulers
from torch_geometric.seed import seed_everything
from .utils import ParameterKeys
from torch.utils.data import Dataset, DataLoader
import os
import torch


class TrainingRun(Run):
    def init(self):
        print("Init generals...")
        self._init_general()
        print("Done!")
        print("Init Dataloaders...")
        self._init_loaders()
        print("Done!")
        print("Init Model...")
        self._init_model()
        print("Done!")
        print("Init Optimizer...")
        self._init_optimizer()
        print("Done!")
        print("Init early stop...")
        self._init_early_stop()
        print("Done!")
        print("Init scheduler...")
        self._init_scheduler()
        print("Done!")
        print("Init Criterion...")
        self._init_criterion()
        print("Done!")
        print("Init metrics...")
        self._init_metrics()
        print("Done!")

    def _init_general(self):
        general_parameters = self.parameters.get(ParameterKeys.GENERAL, dict())
        self.out_dir = general_parameters.get(ParameterKeys.OUT_DIR, "./")
        os.makedirs(self.out_dir, exist_ok=True)
        self.pbar = general_parameters.get(ParameterKeys.PBAR, False)
        self.num_epochs = general_parameters.get(ParameterKeys.NUM_EPOCHS, 1)
        self.device = general_parameters.get(ParameterKeys.DEVICE, "cpu")
        self.seed = general_parameters.get(ParameterKeys.SEED, None)
        seed_everything(self.seed)

    def _init_dataset(self, dataset_params: dict = None) -> Dataset:
        if dataset_params is None:
            return None
        dataset_name = dataset_params.get(ParameterKeys.NAME)
        dataset_cfg = dataset_params.get(ParameterKeys.CFG, dict())
        return data_pkg.__dict__[dataset_name](**dataset_cfg)

    def _init_loaders(self):
        dataset_parameters = self.parameters.get(ParameterKeys.DATA)
        train_dataset, val_dataset, test_dataset = (
            self._init_dataset(dataset_parameters.get(ParameterKeys.TRAIN, None)),
            self._init_dataset(dataset_parameters.get(ParameterKeys.VAL, None)),
            self._init_dataset(dataset_parameters.get(ParameterKeys.TEST, None)),
        )
        loader_parameters = self.parameters.get(ParameterKeys.LOADER, dict())
        self.train_loader = DataLoader(dataset=train_dataset, **loader_parameters)
        self.val_loader = DataLoader(dataset=val_dataset, **loader_parameters)
        self.test_loader = (
            None
            if test_dataset is None
            else DataLoader(dataset=test_dataset, **loader_parameters)
        )

    def _init_model(self):
        model_parameters = self.parameters.get(ParameterKeys.MODEL)
        model_name = model_parameters.get(ParameterKeys.NAME)
        model_config = model_parameters.get(ParameterKeys.CFG, dict())
        self.model = model_pkg.__dict__[model_name](**model_config).to(self.device)

    def _init_criterion(self):
        criterion_parameters = self.parameters.get(ParameterKeys.CRITERION, None)
        criterion_name = criterion_parameters.get(ParameterKeys.NAME)
        criterion_cfg = criterion_parameters.get(ParameterKeys.CFG, {})
        self.criterion = loss_pkg.__dict__[criterion_name](**criterion_cfg)

    def _init_cosine_scheduler_params(self, params) -> schedulers.LambdaLR:
        # get total steps
        total_steps = self.num_epochs * len(self.train_loader)
        params.update({"num_training_steps": total_steps})
        return params

    def _init_scheduler(self):
        scheduler_parameters = self.parameters.get(ParameterKeys.SCHEDULER, None)
        scheduler_name = scheduler_parameters.get(ParameterKeys.NAME)
        scheduler_config = scheduler_parameters.get(ParameterKeys.CFG, {})
        if scheduler_name == ParameterKeys.HUGGINGFACE_COS_SCHEDULER:
            scheduler_config = self._init_cosine_scheduler_params(
                params=scheduler_config
            )
            self.cosine_sched = True
        else:
            self.cosine_sched = False
        self.scheduler = scheduler_registry[scheduler_name](
            optimizer=self.optimizer, **scheduler_config
        )

    def _init_optimizer(self):
        optimizer_prameters = self.parameters.get(ParameterKeys.OPTIMIZER, dict())
        optimizer_name = optimizer_prameters.get(ParameterKeys.NAME)
        optimizer_cfg = optimizer_prameters.get(ParameterKeys.CFG, dict())
        self.optimizer = torch.optim.__dict__[optimizer_name](
            params=self.model.parameters(), **optimizer_cfg
        )

    def _init_early_stop(self):
        early_stop_parameters = self.parameters.get(ParameterKeys.EARLY_STOP, None)
        early_stop_name = early_stop_parameters.get(ParameterKeys.NAME)
        early_stop_cfg = early_stop_parameters.get(ParameterKeys.CFG, {})
        early_stop_cfg[ParameterKeys.BASE_PATH] = self.out_dir
        self.early_stop = early_stop_pkg.__dict__[early_stop_name](**early_stop_cfg)

    def _init_metrics(self):
        metrics_parameters = self.parameters.get(ParameterKeys.METRICS, {})
        self.metrics = MetricCollection(
            {
                k: metrics_pkg.__dict__[v.get(ParameterKeys.NAME)](
                    **v.get(ParameterKeys.CFG, {})
                )
                for k, v in metrics_parameters.items()
            }
        )

    def train_epoch(self, epoch: int):
        raise NotImplementedError()

    def val_epoch(self, epoch: int):
        raise NotImplementedError()

    def test(self):
        raise NotImplementedError()

    def launch(self):
        print("Start experiment!")
        self.model = self.model.to(self.device)
        for epoch in range(1, self.num_epochs + 1):
            self.train_epoch(epoch=epoch)
            self.val_epoch(epoch=epoch)
            if self.trigger:
                print(f"Early stopping at epoch {epoch}/{self.num_epochs}")
                break
        return self.test()
