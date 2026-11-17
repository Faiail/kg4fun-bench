from torch.utils.data import Dataset
import pandas as pd
from src.utils import load_json
from src.data.utils import BatchKeys
from src.data.text.template_keys import TemplateKeys


class FeatureTextDataset(Dataset):
    def __init__(
        self,
        node_factory: str,
        properties: str,
        qid_colname: str,
    ):
        super().__init__()
        self.node_df = pd.read_csv(node_factory, index_col=0)
        self.properties = load_json(properties)
        self.qid_colname = qid_colname

    def __len__(self):
        return len(self.node_df)

    def __getitem__(self, index):
        name = self.node_df.iloc[index][self.qid_colname]
        name_str = str(name)
        entity = (
            self.properties.get(name_str)
            or self.properties.get(f"class_{name_str}")
            or self.properties.get(f"rel_{name_str}")
            or {}
        )
        return f"{TemplateKeys.LABEL} {entity.get(BatchKeys.LABEL, '-')}. {TemplateKeys.DESC} {entity.get(BatchKeys.DESCRIPTION, '-') }."
    
