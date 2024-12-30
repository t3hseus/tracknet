import torch
from typing import List, TypedDict
from .dataset import Track


class BatchSample(TypedDict):
    inputs: torch.Tensor
    targets: torch.Tensor
    lengths: List[int]


def collate_fn(batch: List[Track]) -> BatchSample:
    """
    Collate function for batching Track objects for TrackNET training,
    which predicts the next hit location given the previous hits.

    Args:
        batch (List[Track]): A list of Track objects to be batched.

    Returns:
        BatchSample: A dictionary containing:
            - inputs (torch.Tensor): Zero-padded tensor of shape (batch_size, max_seq_len, 3) 
              containing the input sequences.
            - targets (torch.Tensor): Zero-padded tensor of shape (batch_size, max_seq_len, 3) 
              containing the target sequences.
            - lengths (List[int]): A list of integers indicating the number of valid steps 
              in each sequence.

    Steps:
    1) Identify the maximum (N-1) length among tracks in this batch.
    2) Create zero-padded tensors:
       - input_X:  shape (batch_size, max_seq_len, 3)
       - target_Y: shape (batch_size, max_seq_len, 3)
         where 3 = number of coordinates per hit (x, y, z).
    3) Populate these tensors with the shifted sequences from each track.
    4) Return (input_X, target_Y, lengths), 
       where lengths tells you how many valid steps each sequence has.
    """
    # 1) The max number of (input) steps = (max hits in track) - 1
    max_len = max(len(track.hits_xyz) for track in batch) - 1
    batch_size = len(batch)

    # 2) Prepare zero-padded tensors
    input_X = torch.zeros(batch_size, max_len, 3, dtype=torch.float32)
    target_Y = torch.zeros(batch_size, max_len, 3, dtype=torch.float32)
    lengths = []

    # 3) Fill the tensors
    for i, track in enumerate(batch):
        hits = track.hits_xyz  # shape (N, 3)
        n = len(hits) - 1      # number of training steps for this track
        # input: hits[:-1], target: hits[1:]
        input_X[i, :n, :] = torch.tensor(hits[:-1], dtype=torch.float32)
        target_Y[i, :n, :] = torch.tensor(hits[1:],  dtype=torch.float32)
        lengths.append(n)

    # 4) Return batched data
    return BatchSample(inputs=input_X, targets=target_Y, lengths=lengths)
