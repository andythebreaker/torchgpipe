import pytest
import torch
from torch import nn

from torchgpipe import GPipe


def count_grad_fn(grad_fn, name, visited=None):
    if grad_fn is None:
        return 0

    if visited is None:
        visited = set()
    if grad_fn in visited:
        return 0
    visited.add(grad_fn)

    counter = 0
    if grad_fn.__class__.__name__ == name:
        counter += 1

    for next_grad_fn, _ in grad_fn.next_functions:
        counter += count_grad_fn(next_grad_fn, name, visited=visited)

    return counter


def build_model(layers=3):
    return nn.Sequential(*(nn.Linear(1, 1) for _ in range(layers)))


def test_checkpoint_layers_default_still_uses_partition_boundaries():
    model = GPipe(build_model(), balance=[3], devices=['cpu'], chunks=2,
                  checkpoint='always')

    output = model(torch.rand(2, 1))

    assert model.checkpoint_layers is None
    assert count_grad_fn(output.grad_fn, 'CheckpointBackward') == 2


def test_checkpoint_layers_adds_boundaries_inside_partition():
    model = GPipe(build_model(), balance=[3], devices=['cpu'], chunks=2,
                  checkpoint='always', checkpoint_layers=[1, 2])

    output = model(torch.rand(2, 1))

    assert model.checkpoint_layers == [1, 2]
    assert count_grad_fn(output.grad_fn, 'CheckpointBackward') == 6


def test_checkpoint_layers_with_except_last():
    model = GPipe(build_model(), balance=[3], devices=['cpu'], chunks=3,
                  checkpoint='except_last', checkpoint_layers=1)

    output = model(torch.rand(3, 1))

    assert model.checkpoint_layers == [1]
    assert count_grad_fn(output.grad_fn, 'CheckpointBackward') == 4


def test_checkpoint_layers_supports_negative_indices():
    model = GPipe(build_model(), balance=[3], devices=['cpu'], chunks=2,
                  checkpoint='always', checkpoint_layers=-1)

    output = model(torch.rand(2, 1))

    assert model.checkpoint_layers == [2]
    assert count_grad_fn(output.grad_fn, 'CheckpointBackward') == 4


def test_checkpoint_layers_works_across_partitions():
    model = GPipe(build_model(), balance=[2, 1], devices=['cpu', 'cpu'], chunks=2,
                  checkpoint='always', checkpoint_layers=1)

    output = model(torch.rand(2, 1))

    assert model.checkpoint_layers == [1]
    assert count_grad_fn(output.grad_fn, 'CheckpointBackward') == 6


def test_checkpoint_layers_empty_iterable_keeps_partition_boundaries():
    model = GPipe(build_model(), balance=[3], devices=['cpu'], chunks=2,
                  checkpoint='always', checkpoint_layers=[])

    output = model(torch.rand(2, 1))

    assert model.checkpoint_layers == []
    assert count_grad_fn(output.grad_fn, 'CheckpointBackward') == 2


def test_checkpoint_layers_invalid_index():
    with pytest.raises(ValueError, match='checkpoint layer index out of range'):
        GPipe(build_model(), balance=[3], devices=['cpu'], checkpoint_layers=3)


@pytest.mark.parametrize('checkpoint_layers', [1.5, [1.5]])
def test_checkpoint_layers_requires_integer_indices(checkpoint_layers):
    with pytest.raises(TypeError, match='checkpoint_layers'):
        GPipe(build_model(), balance=[3], devices=['cpu'], checkpoint_layers=checkpoint_layers)
