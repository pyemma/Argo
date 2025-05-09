from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class ModelConfig:
    user_cardinality: int
    item_cardinality: int
    user_embedding_dim: int
    item_embedding_dim: int
    num_dense_features: int


DIN_SMALL_CONFIG = ModelConfig(
    user_cardinality=2000,
    item_cardinality=200000,
    user_embedding_dim=8,
    item_embedding_dim=8,  # for now this is not customizable yet
    num_dense_features=8,
)


@dataclass
class DCNv2Config:
    feature_config: Dict[str, Tuple[int, int]]  # a dict contains the cardinality and embedding dim of each feature
    num_cross_layers: int  # number of layers in cross net
    deep_net_hidden_dims: List[int]  # hidden dimensions of linear layers in deep net
    head_hidden_dim: int  # hidden dimension of prediction head
    input_dim: Optional[int] = None  # input dimension of DCNv2 in case we don't need the embedding layer, e.g. TransAct

    def __post_init__(self):
        if self.feature_config and self.input_dim:
            raise ValueError("Could not set both `feature_config` and `input_dim` field")


@dataclass
class TransActModuleConfig:
    max_seq_len: int  # max length of user behavior sequence
    num_action: int  # number of actions
    action_emb_dim: int  # dimension of action embedding
    item_emb_dim: int  # dimension of item embedding
    top_k: int  # transformer output compression, preserver the first k columns
    transformer_num_head: int  # number of head in transformer block
    transformer_hidden_dim: int  # hidden dimension of transformer block
    num_transformer_block: int  # number of transformer block layers

    def transact_out_dim(self):
        """
            A helper function to return the output dimension of transact module
        """
        return (self.top_k + 1) * (self.action_emb_dim + 2 * self.item_emb_dim)


@dataclass
class TransActModelConfig:
    transact_module_config: TransActModuleConfig
    dcnv2_config: DCNv2Config