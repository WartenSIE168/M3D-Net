import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.optim import AdamW
from model import StableCNNLSTM
from config import device

def configure_optimizer(model, max_epochs=50, steps_per_epoch=100):
    base_model = model.module if isinstance(model, torch.nn.DataParallel) else model

    param_groups = [
        {'params': base_model.img_cnn.parameters(), 'lr': 3e-5},
        {'params': base_model.eye_txt_bilstm.parameters(), 'lr': 1e-4},
        {'params': base_model.flight_txt_bilstm.parameters(), 'lr': 1e-4},
        {'params': base_model.cross_attn.parameters(), 'lr': 3e-4},
        {'params': base_model.convlstm.parameters(), 'lr': 3e-4},
        {'params': base_model.classifier.parameters(), 'lr': 6e-4}
    ]
    optimizer = AdamW(param_groups, weight_decay=1e-4)

    total_steps = max_epochs * steps_per_epoch
    scheduler = lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=6e-4,
        total_steps=total_steps,
        pct_start=0.2,
        anneal_strategy='linear'
    )
    return optimizer, scheduler

def find_matching_indices(list1, list2):
    """
    Find the indices of equal elements in two lists.
    Parameters:
        list1 (list): The first list (no duplicate elements).
        list2 (list): The second list.
    Returns:
        list: A list containing tuples, where each tuple is (element, index in list1, index in list2).
    """
    element_to_index = {element: index for index, element in enumerate(list1)}
    result = []

    for index2, element in enumerate(list2):
        if element in element_to_index:
            index1 = element_to_index[element]
            result.append((element, index1, index2))

    return result
    