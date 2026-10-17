# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Models can keep checkpoint tensors from ever being read at load time."""

import builtins
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

import vllm.model_executor.model_loader.default_loader as default_loader
import vllm.model_executor.model_loader.weight_utils as weight_utils
from vllm.config.load import LoadConfig
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader

_SKIPPED_BYTES = 1 << 20


def _write_checkpoint(path) -> None:
    """One file holding a large skipped tensor, one with kept tensors only."""
    save_file(
        {
            "a.kept.weight": torch.arange(4, dtype=torch.float32),
            "a.skipped.weight": torch.zeros(_SKIPPED_BYTES, dtype=torch.uint8),
            "a.kept.bias": torch.full((3,), 7, dtype=torch.bfloat16),
        },
        str(path / "model-00001.safetensors"),
        metadata={"format": "pt"},
    )
    save_file({"b.kept.weight": torch.ones(2)}, str(path / "model-00002.safetensors"))


def _model(require_skip: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        should_skip_checkpoint_weight=lambda name: ".skipped." in name,
        requires_checkpoint_weight_skip=require_skip,
    )


def _load(path, load_config: LoadConfig, model) -> dict[str, torch.Tensor]:
    loader = DefaultModelLoader(load_config)
    model_config = SimpleNamespace(model=str(path), revision=None)
    return dict(loader.get_all_weights(model_config, model))


def _check_kept(weights: dict[str, torch.Tensor]) -> None:
    assert sorted(weights) == ["a.kept.bias", "a.kept.weight", "b.kept.weight"]
    assert torch.equal(weights["a.kept.weight"], torch.arange(4.0))
    assert torch.equal(
        weights["a.kept.bias"], torch.full((3,), 7, dtype=torch.bfloat16)
    )


@pytest.mark.parametrize(
    "load_config",
    [
        LoadConfig(load_format="safetensors"),
        LoadConfig(safetensors_load_strategy="lazy"),
        LoadConfig(model_loader_extra_config={"enable_multithread_load": True}),
    ],
    ids=["default", "lazy", "multithread"],
)
def test_mmap_loads_never_materialize_skipped_weights(
    tmp_path, monkeypatch, load_config
):
    _write_checkpoint(tmp_path)
    read: list[str] = []
    real_safe_open = weight_utils.safe_open

    class SpySafeOpen:
        def __init__(self, *args, **kwargs):
            self._file = real_safe_open(*args, **kwargs)

        def __enter__(self):
            file = self._file.__enter__()
            real_get_tensor = file.get_tensor

            def get_tensor(name: str) -> torch.Tensor:
                read.append(name)
                return real_get_tensor(name)

            return SimpleNamespace(keys=file.keys, get_tensor=get_tensor)

        def __exit__(self, *exc):
            return self._file.__exit__(*exc)

    real_load_file = weight_utils.load_file

    def spy_load_file(*args, **kwargs) -> dict[str, torch.Tensor]:
        tensors = real_load_file(*args, **kwargs)
        read.extend(tensors)
        return tensors

    monkeypatch.setattr(weight_utils, "safe_open", SpySafeOpen)
    monkeypatch.setattr(weight_utils, "load_file", spy_load_file)

    _check_kept(_load(tmp_path, load_config, _model()))
    assert not any(".skipped." in name for name in read)


def test_eager_load_reads_only_kept_bytes(tmp_path, monkeypatch):
    """Eager loading must not pull a whole file holding a skipped table."""
    _write_checkpoint(tmp_path)
    bytes_read = 0

    class CountingFile:
        def __init__(self, file):
            self._file = file

        def read(self, *args):
            nonlocal bytes_read
            data = self._file.read(*args)
            bytes_read += len(data)
            return data

        def __getattr__(self, name):
            return getattr(self._file, name)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return self._file.__exit__(*exc)

    monkeypatch.setattr(
        weight_utils,
        "open",
        lambda *args, **kwargs: CountingFile(
            builtins.open(*args, **kwargs)  # noqa: SIM115 closed by the loader
        ),
        raising=False,
    )

    weights = _load(tmp_path, LoadConfig(safetensors_load_strategy="eager"), _model())

    _check_kept(weights)
    assert bytes_read < _SKIPPED_BYTES


def test_prefetch_leaves_out_files_with_skipped_weights(tmp_path, monkeypatch):
    _write_checkpoint(tmp_path)
    prefetched: list[str] = []
    monkeypatch.setattr(
        weight_utils,
        "_prefetch_all_checkpoints",
        lambda files, **kwargs: prefetched.extend(files),
    )

    weights = _load(
        tmp_path, LoadConfig(safetensors_load_strategy="prefetch"), _model()
    )

    _check_kept(weights)
    assert [path.rsplit("/", 1)[-1] for path in prefetched] == [
        "model-00002.safetensors"
    ]


@pytest.mark.parametrize("load_format", ["fastsafetensors", "instanttensor"])
def test_required_skip_rejects_formats_that_read_everything(
    tmp_path, monkeypatch, load_format
):
    _write_checkpoint(tmp_path)
    monkeypatch.setattr(
        default_loader,
        f"{load_format}_weights_iterator",
        lambda *args, **kwargs: pytest.fail("the checkpoint was read"),
    )
    loader = DefaultModelLoader(LoadConfig(load_format=load_format))
    model_config = SimpleNamespace(model=str(tmp_path), revision=None)

    with pytest.raises(ValueError, match="must leave some unread"):
        next(loader.get_all_weights(model_config, _model(require_skip=True)))
