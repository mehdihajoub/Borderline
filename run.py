import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.optim import lr_scheduler
import torch.utils.data as data
from torchvision import transforms
from transformers import BertTokenizer, AdamW

import numpy as np
import os
import time
import copy
import random

from classifier_dataset import TweetsDataset




if __name__ == 'main' :

    #HyperParameter
    max_len = 128
    batch_size = 32
    lr = 2e-5
    vocab_size = 30522
    embed_size = 768
    num_heads =  12
    ff_hidden_size = 3072

    csv_path = 'data/labeled_data.csv'
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    dataset = TweetsDataset(csv_path,tokenizer, )

