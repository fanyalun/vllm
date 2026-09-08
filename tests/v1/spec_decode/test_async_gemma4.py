# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch
from safetensors.torch import save_file
from torch import nn
from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig

from vllm.transformers_utils.model_arch_config_convertor import (
    Gemma4MTPModelArchConfigConvertor,
)
from vllm.v1.worker.gpu.spec_decode.async_draft.gemma4 import (
    Gemma4Draft,
    pack_target_kv,
    target_layer_indices,
)
from vllm.v1.worker.gpu.spec_decode.async_draft.weights import (
    materialize_gemma4_embedding,
)


def test_target_layers_exclude_shared_tail():
    text = SimpleNamespace(
        layer_types=["sliding", "full", "sliding", "full"], num_kv_shared_layers=2
    )
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(get_text_config=lambda: text)
        )
    )
    assert target_layer_indices(config) == {"sliding": 0, "full": 1}


def test_assistant_heterogeneous_dimensions_and_feedback_width():
    text = Gemma4TextConfig(
        num_hidden_layers=4,
        hidden_size=1024,
        num_attention_heads=16,
        num_key_value_heads=8,
        head_dim=256,
        global_head_dim=512,
        num_global_key_value_heads=2,
        attention_k_eq_v=True,
        layer_types=["sliding_attention"] * 3 + ["full_attention"],
    )
    converter = Gemma4MTPModelArchConfigConvertor(
        SimpleNamespace(backbone_hidden_size=2816), text
    )
    assert converter.get_hidden_size() == 2816
    assert converter.get_head_size() == 512
    assert converter.get_total_num_kv_heads() == 8


def test_kv_pack_crosses_pages_and_is_independent():
    cache = torch.arange(5 * 2 * 4 * 2 * 3).reshape(5, 2, 4, 2, 3)
    table = torch.tensor([3, 1, 4])
    packed = pack_target_kv(cache, table, 10)
    expected = torch.stack([cache[table[i // 4], :, i % 4] for i in range(10)])
    torch.testing.assert_close(packed, expected)
    cache.zero_()
    torch.testing.assert_close(packed, expected)
    destination = torch.zeros(3, 4, 2, 2, 3, dtype=packed.dtype).transpose(1, 2)
    view = destination.transpose(1, 2).reshape(-1, 2, 2, 3)
    view[:10].copy_(packed)
    torch.testing.assert_close(destination[1, :, 0], packed[4])


def test_target_embedding_preserves_assistant_lm_head(tmp_path, monkeypatch):
    from vllm.model_executor.layers import vocab_parallel_embedding

    monkeypatch.setattr(
        vocab_parallel_embedding, "VocabParallelEmbedding", nn.Embedding
    )
    target = torch.arange(40, dtype=torch.float32).view(10, 4)
    save_file(
        {"model.language_model.embed_tokens.weight": target},
        tmp_path / "model.safetensors",
    )
    model = nn.Module()
    model.model = nn.Module()
    model.model.embed_tokens = nn.Embedding(10, 2)
    model.model.backbone_hidden_size = 4
    model.model.vocab_size = 10
    model.lm_head = nn.Linear(2, 10, bias=False)
    model.lm_head.weight = model.model.embed_tokens.weight
    model.has_own_lm_head = True
    original_head = model.lm_head.weight
    values = original_head.detach().clone()
    audit = materialize_gemma4_embedding(model, str(tmp_path))
    assert model.lm_head.weight is original_head
    torch.testing.assert_close(model.lm_head.weight, values)
    torch.testing.assert_close(model.model.embed_tokens.weight, target)
    assert [row["source"] for row in audit] == ["target", "draft"]


def test_decode_keeps_positions_and_snapshot_fixed():
    draft = object.__new__(Gemma4Draft)
    calls = []
    snapshot = {"full": torch.zeros(1)}

    def forward(ids, positions, hidden, snapshots, seq_len, query_len):
        calls.append((positions.clone(), snapshots, seq_len, hidden.clone()))
        return hidden, hidden + 1

    draft.forward = forward
    draft.model = SimpleNamespace(compute_logits=lambda hidden: hidden)
    ids = torch.tensor([2, 3])
    pos = torch.tensor([19, 19])
    tokens, _, _ = draft.decode(ids, pos, torch.zeros(2, 5), snapshot, 23, 6)
    assert tokens.shape == (2, 6)
    for i, (position, kv, length, hidden) in enumerate(calls):
        assert kv is snapshot and length == 23
        torch.testing.assert_close(position, pos)
        torch.testing.assert_close(hidden, torch.full((2, 5), float(i)))


def test_fanout_covers_bonus_and_replays_returned_prefix(monkeypatch):
    from vllm.v1.worker.gpu.spec_decode.async_draft.cache import BranchCache

    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args: None)
    monkeypatch.setattr(
        torch.cuda, "Event", lambda: SimpleNamespace(record=lambda: None)
    )
    draft = object.__new__(Gemma4Draft)
    draft.width, draft.fan_out, draft.device = 6, 3, "cpu"
    scores = torch.arange(12, dtype=torch.float32).view(1, -1)
    draft.fresh = lambda *args, **kwargs: (
        None,
        [scores],
        [torch.ones(1, 2)],
        torch.tensor([10]),
        11,
    )
    replayed = []
    draft.forward = lambda ids, *args, **kwargs: (
        replayed.append(ids.item()) or scores,
        torch.ones(1, 2),
    )
    draft.model = SimpleNamespace(compute_logits=lambda h: h)
    branch_positions = []

    def decode(ids, positions, *args):
        branch_positions.append(positions.clone())
        return ids[:, None].expand(-1, 6), None, None

    draft.decode = decode
    slot = SimpleNamespace(target_kv={})
    batch = SimpleNamespace(engine_instance_id="e", req_ids=["r"], request_epochs=[2])
    cache = BranchCache()
    returned = torch.tensor([[11, 10, 9, 8, 7, 6]])
    draft.build(cache, slot, batch, returned)
    assert replayed == returned[0].tolist()
    assert len(cache.entries) == 21
    for depth, positions in enumerate(branch_positions):
        torch.testing.assert_close(positions, torch.full((3,), 11 + depth))
    assert {key[3] for key in cache.entries} == set(range(7))
    for key, branch in cache.entries.items():
        assert key[:3] == ("e", "r", 2)
        if key[3] < 6:
            assert key[4] != returned[0, key[3]]
        assert branch.provisional_state is None
        assert branch.tokens.shape == (1, 6)
