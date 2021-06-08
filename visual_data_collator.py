import random
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
import os
import pickle
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from time import time
from transformers.tokenization_utils_base import BatchEncoding

InputDataClass = NewType("InputDataClass", Any)
DataCollator = NewType("DataCollator", Callable[[List[InputDataClass]], Dict[str, torch.Tensor]])

dataroot = 'data'
all_imgid2idx_f = os.path.join(dataroot, 'all_imgid2idx.pkl')
all_img_features_f = os.path.join(dataroot, 'img36_all.npy')
with open(all_imgid2idx_f, 'rb') as f:
    all_imgid2idx = pickle.load(f)
    
all_img_features = np.load(all_img_features_f)

def visual_data_collator(features: List[InputDataClass]) -> Dict[str, torch.Tensor]:
    """
    Very simple data collator that simply collates batches of dict-like objects and performs special handling for
    potential keys named:
        - ``label``: handles a single value (int or float) per object
        - ``label_ids``: handles a list of values per object
    Does not do any additional preprocessing: property names of the input object will be used as corresponding inputs
    to the model. See glue and ner for example of how it's useful.
    """

    # In this function we'll make the assumption that all `features` in the batch
    # have the same attributes.
    # So we will look at the first element as a proxy for what attributes exist
    # on the whole batch.

    if not isinstance(features[0], (dict, BatchEncoding)):
        features = [vars(f) for f in features]

    first = features[0]
    batch = {}

    # Special handling for labels.
    # Ensure that tensor is created with the correct type
    # (it should be automatically the case, but let's make sure of it.)
    if "label" in first and first["label"] is not None:
        label = first["label"].item() if isinstance(first["label"], torch.Tensor) else first["label"]
        dtype = torch.long if isinstance(label, int) else torch.float
        batch["labels"] = torch.tensor([f["label"] for f in features], dtype=dtype)
    elif "label_ids" in first and first["label_ids"] is not None:
        if isinstance(first["label_ids"], torch.Tensor):
            batch["labels"] = torch.stack([f["label_ids"] for f in features])
        else:
            dtype = torch.long if type(first["label_ids"][0]) is int else torch.float
            batch["labels"] = torch.tensor([f["label_ids"] for f in features], dtype=dtype)


    # Handling of all other possible keys.
    # Again, we will use the first element to figure out which key/values are not None for this model.
    for k, v in first.items():
        if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
            prev_t = time()

            if (k == 'input_ids'):
                batch_input_ids = torch.tensor([f[k] for f in features])
                batch_size = batch_input_ids.size(0)
                img_prefix_token = torch.tensor([1023,10]).repeat(batch_size, 1)
                batch[k] = torch.cat((batch_input_ids, img_prefix_token), axis=1)
                
            elif(k == 'attention_mask'):
                batch_att_mask = torch.tensor([f[k] for f in features])
                batch_size = batch_att_mask.size(0)
                additional_att_mask = torch.tensor([1]*38).repeat(batch_size, 1)
                batch[k] = torch.cat((batch_att_mask, additional_att_mask), axis=1)

            elif(k == 'img_features'):
                batch[k] = torch.stack([torch.from_numpy(all_img_features[all_imgid2idx[f[k]]]) for f in features])

            elif isinstance(v, torch.Tensor):
                batch[k] = torch.stack([f[k] for f in features])
            else:
                batch[k] = torch.tensor([f[k] for f in features])

    return batch
