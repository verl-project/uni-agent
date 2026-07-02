"""CPU-only unit tests for the gateway prefix trie (issue #51, M1).

These tests exercise the trie's structural behavior directly with synthetic
token buffers — no tokenizer, model, or Ray actor — so they run on CPU. They
cover the fork types from the RFC appendix:

- A. sequential extension       (single chain, prefix reuse)
- B. system-keyed split         (parallel sub-agents)
- C. context condensation       (sibling branch off a fork point)
- D. best-of-N / idempotent retry
- E. warm-start                 (seeded no-checkpoint nodes)

plus the core mechanics: pending nodes, checkpoint cloning, MessageKey
canonicalization, and multimodal digest keying.
"""

from __future__ import annotations

import pytest

from uni_agent.gateway.session.trie import (
    PrefixTrie,
    canonicalize_content,
    make_message_key,
)
from uni_agent.gateway.session.types import TrajectoryBuffer

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def sys_msg(text="you are a coding agent"):
    return {"role": "system", "content": text}


def user_msg(text):
    return {"role": "user", "content": text}


def asst_msg(text, tool_calls=None):
    msg = {"role": "assistant", "content": text}
    if tool_calls is not None:
        msg["tool_calls"] = tool_calls
    return msg


def tool_msg(text, tool_call_id="call0"):
    return {"role": "tool", "content": text, "tool_call_id": tool_call_id}


def make_buffer(prompt_len, response_tokens, mask_value=1):
    """A synthetic buffer standing in for a real tokenization."""
    return TrajectoryBuffer(
        prompt_ids=list(range(prompt_len)),
        response_ids=list(response_tokens),
        response_mask=[mask_value] * len(response_tokens),
        response_logprobs=[0.0] * len(response_tokens),
    )


def commit_turn(trie, messages, assistant_msg, buffer, *, incremental=None, **kw):
    """Run one prepare -> commit cycle and return (prepare_result, node).

    ``incremental`` mirrors the production ``use_incremental`` signal that the
    gateway passes to ``commit``. When left as ``None`` it is derived the same way
    the gateway derives it for the tool-less unit path: a reused checkpoint
    (``trajectory_buffer is not None``) means this turn extended an existing
    branch incrementally. Pass it explicitly to model a full re-encode that reused
    a prefix node but not its tokens (e.g. a mid-session tools change)."""
    pr = trie.prepare(messages)
    if incremental is None:
        incremental = pr.trajectory_buffer is not None
    node = trie.commit(pr.branch_handle, buffer, assistant_msg, messages=messages, incremental=incremental, **kw)
    return pr, node


# ---------------------------------------------------------------------------
# MessageKey canonicalization
# ---------------------------------------------------------------------------


def test_message_key_is_hashable_and_stable():
    k1 = make_message_key(user_msg("hi"))
    k2 = make_message_key(user_msg("hi"))
    assert k1 == k2
    assert hash(k1) == hash(k2)
    assert k1 != make_message_key(user_msg("bye"))


def test_tool_call_argument_canonicalization_matches():
    # Same logical tool call, different JSON string formatting -> same key.
    a = asst_msg("", tool_calls=[{"id": "c1", "function": {"name": "f", "arguments": '{"a": 1, "b": 2}'}}])
    b = asst_msg("", tool_calls=[{"id": "c1", "function": {"name": "f", "arguments": '{"b":2,"a":1}'}}])
    assert make_message_key(a) == make_message_key(b)


def test_reasoning_content_excluded_from_key():
    a = {"role": "assistant", "content": "answer", "reasoning_content": "think A"}
    b = {"role": "assistant", "content": "answer", "reasoning_content": "think B"}
    assert make_message_key(a) == make_message_key(b)


def test_multimodal_content_digest_keying():
    img_a = user_msg([{"type": "image_url", "image_url": {"url": "http://x/a.png"}}, {"type": "text", "text": "what"}])
    img_a2 = user_msg([{"type": "image_url", "image_url": {"url": "http://x/a.png"}}, {"type": "text", "text": "what"}])
    img_b = user_msg([{"type": "image_url", "image_url": {"url": "http://x/b.png"}}, {"type": "text", "text": "what"}])
    assert make_message_key(img_a) == make_message_key(img_a2)
    assert make_message_key(img_a) != make_message_key(img_b)


def test_canonicalize_content_shapes():
    assert canonicalize_content("plain") == "plain"
    assert canonicalize_content(None) is None
    parts = canonicalize_content([{"type": "text", "text": "hi"}])
    assert parts == (("text", "hi"),)
    assert isinstance(make_message_key(user_msg([{"type": "text", "text": "hi"}])).content, tuple)


# ---------------------------------------------------------------------------
# A. sequential extension — single chain, prefix reuse
# ---------------------------------------------------------------------------


def test_sequential_extension_single_chain():
    trie = PrefixTrie()
    msgs = [sys_msg(), user_msg("fix bug")]

    # turn 1: first call, no checkpoint -> full encode path
    pr1 = trie.prepare(msgs)
    assert pr1.trajectory_buffer is None
    assert pr1.checkpoint_messages == []
    a1 = asst_msg("looking", tool_calls=[{"id": "c1", "function": {"name": "cat", "arguments": "{}"}}])
    trie.commit(pr1.branch_handle, make_buffer(100, [200, 201, 202]), a1, messages=msgs)

    # turn 2: append tool result; must match prefix and clone a1's checkpoint
    msgs2 = msgs + [a1, tool_msg("def login(): ...")]
    pr2 = trie.prepare(msgs2)
    assert pr2.trajectory_buffer is not None, "turn 2 should reuse turn 1 checkpoint"
    # checkpoint covers up to and including a1
    assert pr2.checkpoint_messages == [sys_msg(), user_msg("fix bug"), a1]
    a2 = asst_msg("done")
    # turn 2 extends turn 1's cloned checkpoint incrementally -> absorbs a1.
    trie.commit(pr2.branch_handle, make_buffer(100, [200, 201, 202, 50, 51, 300]), a2, messages=msgs2, incremental=True)

    # single linear chain, one exportable branch
    assert trie.num_branches() == 1
    assert trie.num_inflight() == 0


def test_prepare_returns_independent_clone():
    trie = PrefixTrie()
    msgs = [sys_msg(), user_msg("q")]
    a1 = asst_msg("a")
    trie.commit(trie.prepare(msgs).branch_handle, make_buffer(10, [1, 2, 3]), a1, messages=msgs)

    pr = trie.prepare(msgs + [a1, tool_msg("obs")])
    pr.trajectory_buffer.response_ids.append(999)  # mutate the clone
    # stored checkpoint must be unaffected
    node, _ = trie.match(msgs + [a1])
    assert 999 not in node.checkpoint.trajectory_buffer.response_ids


# ---------------------------------------------------------------------------
# B. system-keyed split — parallel sub-agents
# ---------------------------------------------------------------------------


def test_system_keyed_split_parallel_subagents():
    trie = PrefixTrie()
    planner = [sys_msg("you are the planner"), user_msg("task")]
    worker = [sys_msg("you are the worker"), user_msg("task")]

    commit_turn(trie, planner, asst_msg("plan"), make_buffer(80, [1, 2]))
    commit_turn(trie, worker, asst_msg("work"), make_buffer(80, [3, 4]))

    # different system prompts diverge right at the root
    assert len(trie.root.children) == 2
    assert trie.num_branches() == 2


# ---------------------------------------------------------------------------
# C. context condensation — sibling branch off a fork point
# ---------------------------------------------------------------------------


def test_context_condensation_creates_sibling():
    trie = PrefixTrie()
    base = [sys_msg(), user_msg("long task")]
    a1 = asst_msg("step 1")
    commit_turn(trie, base, a1, make_buffer(120, [10, 11, 12]))

    # original continuation
    cont = base + [a1, user_msg("continue")]
    commit_turn(trie, cont, asst_msg("step 2"), make_buffer(120, [10, 11, 12, 20, 21]))

    # condensed continuation: a *different* user message at the same fork point
    condensed = base + [a1, user_msg("[recap] do step 2")]
    pr = trie.prepare(condensed)
    # nearest checkpoint is a1 (the shared prefix), cloned at the splice
    assert pr.trajectory_buffer is not None
    assert pr.checkpoint_messages == base + [a1]
    commit_turn(trie, condensed, asst_msg("step 2 alt"), make_buffer(120, [10, 11, 12, 30, 31]))

    # a1 now has two user children (original + condensed) -> two branches
    a1_node, _ = trie.match(base + [a1])
    assert len(a1_node.children) == 2
    assert trie.num_branches() == 2


# ---------------------------------------------------------------------------
# D. best-of-N and idempotent retry
# ---------------------------------------------------------------------------


def test_best_of_n_siblings_share_pending_parent():
    trie = PrefixTrie()
    msgs = [sys_msg(), user_msg("solve")]

    # three samples of the *same* request -> three prepare calls share the
    # pending user node, then commit three different assistant children.
    handles = [trie.prepare(msgs).branch_handle for _ in range(3)]
    for i, h in enumerate(handles):
        trie.commit(h, make_buffer(60, [i, i + 1]), asst_msg(f"answer {i}"), messages=msgs)

    user_node, _ = trie.match(msgs)
    assert len(user_node.children) == 3, "three distinct assistant siblings"
    assert trie.num_branches() == 3
    assert trie.num_inflight() == 0


def test_idempotent_retry_reuses_node():
    trie = PrefixTrie()
    msgs = [sys_msg(), user_msg("solve")]

    trie.commit(trie.prepare(msgs).branch_handle, make_buffer(60, [1, 2]), asst_msg("same"), messages=msgs)
    trie.commit(trie.prepare(msgs).branch_handle, make_buffer(60, [1, 2]), asst_msg("same"), messages=msgs)

    user_node, _ = trie.match(msgs)
    assert len(user_node.children) == 1, "identical output reuses the node"
    assert trie.num_branches() == 1


# ---------------------------------------------------------------------------
# E. warm-start — seeded no-checkpoint nodes
# ---------------------------------------------------------------------------


def test_warm_start_seeded_nodes_have_no_checkpoint():
    trie = PrefixTrie()
    history = [sys_msg(), user_msg("imported"), asst_msg("imported reply"), user_msg("now continue")]
    # seed the imported transcript as structural nodes (no checkpoints)
    seeded = trie.materialize_prompt_suffix(trie.root, history, 0)
    _, ckpt = trie.nearest_ckpt(seeded)
    assert ckpt is None, "warm-start nodes carry no checkpoint"

    # first live generation from the seeded tail must full-encode (no clone)
    pr = trie.prepare(history)
    assert pr.trajectory_buffer is None
    assert pr.checkpoint_messages == []
    trie.commit(pr.branch_handle, make_buffer(200, [1, 2, 3]), asst_msg("live reply"), messages=history)

    # after the first commit there is a checkpoint to reuse
    assert trie.num_branches() == 1


# ---------------------------------------------------------------------------
# pending nodes / cleanup / export
# ---------------------------------------------------------------------------


def test_pending_node_not_exported_until_commit():
    trie = PrefixTrie()
    msgs = [sys_msg(), user_msg("q")]
    pr = trie.prepare(msgs)
    # a pending user node now exists but no assistant committed beneath it
    assert trie.num_branches() == 0
    assert trie.num_inflight() == 1

    # generation failed -> abandon; node stays but is never exported
    trie.abandon(pr.branch_handle)
    assert trie.num_inflight() == 0
    assert trie.num_branches() == 0
    assert list(trie.iter_export_nodes()) == []


def test_export_emits_terminal_checkpoints_only():
    trie = PrefixTrie()
    msgs = [sys_msg(), user_msg("q")]
    a1 = asst_msg("a1", tool_calls=[{"id": "c1", "function": {"name": "f", "arguments": "{}"}}])
    commit_turn(trie, msgs, a1, make_buffer(50, [1]))
    # continue the branch
    msgs2 = msgs + [a1, tool_msg("obs")]
    commit_turn(trie, msgs2, asst_msg("a2"), make_buffer(50, [1, 2, 3]))

    terminal = list(trie.iter_export_nodes())
    all_nodes = list(trie.iter_export_nodes(export_all=True))
    # default: only the deepest (a2). export_all: both a1 and a2.
    assert len(terminal) == 1
    assert len(all_nodes) == 2


def test_commit_with_unknown_handle_raises():
    trie = PrefixTrie()
    pr = trie.prepare([sys_msg(), user_msg("q")])
    trie.commit(pr.branch_handle, make_buffer(10, [1]), asst_msg("a"), messages=[])
    with pytest.raises(KeyError):
        trie.commit(pr.branch_handle, make_buffer(10, [1]), asst_msg("a"), messages=[])


# ---------------------------------------------------------------------------
# multimodal per-branch reconstruction
# ---------------------------------------------------------------------------


def test_multimodal_collected_per_branch():
    trie = PrefixTrie()
    msgs = [sys_msg(), user_msg([{"type": "image_url", "image_url": {"url": "http://x/a.png"}}])]
    pr = trie.prepare(msgs)
    trie.commit(
        pr.branch_handle,
        make_buffer(300, [1, 2]),
        asst_msg("i see a cat"),
        messages=msgs,
        image_data=["<imgA-features>"],
    )
    node, _ = trie.match(msgs)
    assert node.children, "assistant child committed"
    asst_node = next(iter(node.children.values()))
    images, videos = trie.collect_multi_modal(asst_node)
    assert images == ["<imgA-features>"]
    assert videos is None


# ---------------------------------------------------------------------------
# regression guards (these fail on the pre-review buggy behavior)
# ---------------------------------------------------------------------------


def test_rebuild_messages_isolates_caller_mutation():
    """rebuild_messages must not hand out the trie's own message dicts; mutating
    the result must not corrupt the stored node (fails if references leak)."""
    trie = PrefixTrie()
    msgs = [sys_msg(), user_msg("orig")]
    a1 = asst_msg("a")
    commit_turn(trie, msgs, a1, make_buffer(10, [1]))
    user_node, _ = trie.match(msgs)

    rebuilt = trie.rebuild_messages(user_node)
    rebuilt[0]["role"] = "HACKED"  # caller mutates the returned transcript
    assert user_node.parent.message["role"] == "system", "trie node must be unaffected"


def test_media_digest_hashes_large_data_uri():
    """A long data: URI must be hashed (compact, fixed-width), not stored
    verbatim in the MessageKey (fails if the raw string is kept)."""
    big = "data:image/png;base64," + "A" * 400
    key = make_message_key(user_msg([{"type": "image_url", "image_url": {"url": big}}]))
    _, digest = key.content[0]
    assert digest != big
    assert len(digest) == 16  # sha256[:16]
    # identical URI -> identical key; different URI -> different key
    big2 = "data:image/png;base64," + "B" * 400
    same = make_message_key(user_msg([{"type": "image_url", "image_url": {"url": big}}]))
    other = make_message_key(user_msg([{"type": "image_url", "image_url": {"url": big2}}]))
    assert key == same
    assert key != other


def test_refresh_clears_stale_multimodal():
    """An idempotent refresh whose checkpoint has no multimodal data must clear
    the node's stale image/video (fails if old data lingers)."""
    trie = PrefixTrie()
    msgs = [sys_msg(), user_msg("q")]
    a = asst_msg("same answer")

    trie.commit(trie.prepare(msgs).branch_handle, make_buffer(10, [1]), a, messages=msgs, image_data=["imgA"])
    node, _ = trie.match(msgs)
    asst_node = next(iter(node.children.values()))
    assert asst_node.image_data == ["imgA"]

    # idempotent retry, same output, no images this time
    trie.commit(trie.prepare(msgs).branch_handle, make_buffer(10, [1]), a, messages=msgs, image_data=None)
    assert asst_node.image_data is None, "stale multimodal must be cleared on refresh"


def test_export_terminals_on_deep_branchy_trie():
    """Terminal-checkpoint selection on a deeper, branchy trie — regression
    guard for the O(N) post-order rewrite of iter_export_nodes."""
    trie = PrefixTrie()
    base = [sys_msg(), user_msg("root")]
    a1 = asst_msg("a1", tool_calls=[{"id": "c1", "function": {"name": "f", "arguments": "{}"}}])
    commit_turn(trie, base, a1, make_buffer(10, [1]))
    # continue the a1 branch one more turn (a1 becomes non-terminal)
    cont = base + [a1, tool_msg("obs")]
    commit_turn(trie, cont, asst_msg("a2"), make_buffer(10, [1, 2]))
    # a second sibling under the same user (best-of-N), left as a leaf
    commit_turn(trie, base, asst_msg("a1-alt"), make_buffer(10, [9]))

    terminals = list(trie.iter_export_nodes())
    all_ckpts = list(trie.iter_export_nodes(export_all=True))
    # terminals: a2 (deepest of the a1 branch) + a1-alt leaf -> 2
    assert len(terminals) == 2
    # all checkpoints: a1, a2, a1-alt -> 3
    assert len(all_ckpts) == 3
