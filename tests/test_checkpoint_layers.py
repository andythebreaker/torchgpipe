import pytest
import torch
from torch import nn

from torchgpipe import GPipe


def count_grad_fn(grad_fn, name, visited=None):
    if visited is None:
        visited = set()
    if grad_fn in visited:
        return 0
    visited.add(grad_fn)

    if grad_fn is None:
        return 0
    if grad_fn.__class__.__name__ == name:
        return 1

    counter = 0
    for next_grad_fn, _ in grad_fn.next_functions:
        counter += count_grad_fn(next_grad_fn, name, visited=visited)
    return counter


def model():
    return nn.Sequential(nn.Linear(1, 1),
                         nn.Linear(1, 1),
                         nn.Linear(1, 1),
                         nn.Linear(1, 1))


def checkpoint_backward_count(gpipe):
    output = gpipe(torch.rand(4, 1))
    return count_grad_fn(output.grad_fn, 'CheckpointBackward')


def test_checkpoint_layers_default_preserves_partition_checkpointing():
    gpipe = GPipe(model(),
                  balance=[4],
                  devices=['cpu'],
                  chunks=2,
                  checkpoint='always')

    assert gpipe.checkpoint_layers == [0]
    assert checkpoint_backward_count(gpipe) == 2


def test_checkpoint_layers_adds_checkpoints_within_partition():
    gpipe = GPipe(model(),
                  balance=[4],
                  devices=['cpu'],
                  chunks=2,
                  checkpoint='always',
                  checkpoint_layers=[0, 2])

    assert gpipe.checkpoint_layers == [0, 2]
    assert checkpoint_backward_count(gpipe) == 4


def test_checkpoint_layers_selects_segment_starts():
    gpipe = GPipe(model(),
                  balance=[4],
                  devices=['cpu'],
                  chunks=2,
                  checkpoint='always',
                  checkpoint_layers=[2])

    assert gpipe.checkpoint_layers == [2]
    assert checkpoint_backward_count(gpipe) == 2


def test_checkpoint_layers_empty_disables_checkpointing_segments():
    gpipe = GPipe(model(),
                  balance=[4],
                  devices=['cpu'],
                  chunks=2,
                  checkpoint='always',
                  checkpoint_layers=[])

    assert gpipe.checkpoint_layers == []
    assert checkpoint_backward_count(gpipe) == 0


def test_checkpoint_layers_with_except_last():
    gpipe = GPipe(model(),
                  balance=[4],
                  devices=['cpu'],
                  chunks=2,
                  checkpoint='except_last',
                  checkpoint_layers=[0, 2])

    assert checkpoint_backward_count(gpipe) == 2


def test_checkpoint_layers_negative_index():
    gpipe = GPipe(model(),
                  balance=[4],
                  devices=['cpu'],
                  chunks=2,
                  checkpoint='always',
                  checkpoint_layers=[-1])

    assert gpipe.checkpoint_layers == [3]
    assert checkpoint_backward_count(gpipe) == 2


def test_checkpoint_layers_invalid():
    with pytest.raises(IndexError, match='checkpoint layer index out of range'):
        GPipe(model(),
              balance=[4],
              devices=['cpu'],
              checkpoint_layers=[4])
