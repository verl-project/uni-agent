"""Prefix trie for multi-trajectory session storage (issue #51, M1).

The gateway historically kept a single linear ``active_trajectory`` per session,
so only the latest branch could be re-attached. This module replaces that with a
per-session **prefix trie**: every committed assistant turn becomes a node that
may carry a :class:`BranchCheckpoint`, and an incoming request longest-prefix
matches against any path and continues from the nearest checkpoint.

Design (see ``docs/trie_m1_implementation_plan.md``):

- Each node stores exactly one message; the full transcript is rebuilt by
  walking ``parent`` pointers. Children are keyed by :class:`MessageKey`.
- A generation is ``prepare -> tokenize -> commit``. ``prepare`` walks the trie
  and clones the nearest checkpoint into a request-local buffer; ``commit``
  attaches the assistant child and writes its checkpoint.
- ``prepare`` materializes the incoming prompt-side messages as *pending*
  (structural) nodes that carry no checkpoint and are never exported until an
  assistant child commits beneath them.

This module is intentionally free of the verl-importing codec so its unit tests
run standalone. Tokenization stays in the codec (M1 reuses the existing
``remove_system_prompt`` path); the trie only stores token state, never encodes.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from typing import Any, Iterator

from uni_agent.gateway.session.types import TrajectoryBuffer

# ---------------------------------------------------------------------------
# Message canonicalization -> hashable MessageKey
# ---------------------------------------------------------------------------


def canonicalize_tool_arguments(arguments: Any) -> tuple[str, Any]:
    """Normalize a tool call's ``arguments`` so semantically-equal arguments that
    differ only in JSON string formatting (whitespace, key order) compare equal.

    Mirrors ``codec._canonicalize_tool_arguments_for_comparison`` but is kept
    here so the trie has no dependency on the verl-importing codec module.
    """
    if isinstance(arguments, (dict, list)):
        return ("json", _freeze(arguments))
    if isinstance(arguments, str):
        try:
            return ("json", _freeze(json.loads(arguments)))
        except json.JSONDecodeError:
            return ("raw", arguments)
    return ("raw", arguments)


def _freeze(value: Any) -> Any:
    """Recursively turn dicts/lists into hashable, order-canonical tuples."""
    if isinstance(value, dict):
        return tuple(sorted((k, _freeze(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    return value


def _media_digest(value: Any) -> str:
    """Stable identity digest for an image/video payload.

    Uses the URL/string verbatim when available, otherwise a short sha256 of the
    bytes/representation. Only an identity fingerprint for branch routing — the
    actual pixels are stored on the node, never in the key.
    """
    if isinstance(value, dict):
        # OpenAI-style {"url": ...} / nested {"image_url": {"url": ...}} blocks.
        for key in ("url", "image_url", "video_url", "image", "video", "bytes", "data"):
            if key in value:
                return _media_digest(value[key])
        return hashlib.sha256(repr(_freeze(value)).encode("utf-8")).hexdigest()[:16]
    if isinstance(value, str):
        # Inline ``data:`` URIs (and other very long strings) are hashed so the
        # MessageKey stays compact; short URLs/ids are kept verbatim.
        if value.startswith("data:") or len(value) > 256:
            return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
        return value
    if isinstance(value, (bytes, bytearray)):
        return hashlib.sha256(bytes(value)).hexdigest()[:16]
    # _freeze canonicalizes nested dicts/lists so the repr is order-stable
    # across processes (plain repr of a dict is not guaranteed canonical).
    return hashlib.sha256(repr(_freeze(value)).encode("utf-8")).hexdigest()[:16]


def canonicalize_content(content: Any) -> str | tuple | None:
    """Canonicalize a message ``content`` field into a hashable form.

    Text content passes through as a ``str``; multimodal content (a list of
    parts) becomes a tuple of ``("text", str)`` / ``("image", digest)`` /
    ``("video", digest)`` parts. Image/video payloads are reduced to a stable
    digest so identical media routes to the same node.
    """
    if content is None or isinstance(content, str):
        return content
    if not isinstance(content, list):
        # Unknown scalar shape: freeze for hashability.
        return ("opaque", _freeze(content))

    parts: list[tuple[str, Any]] = []
    for item in content:
        if not isinstance(item, dict):
            parts.append(("text", str(item)))
            continue
        item_type = item.get("type")
        if item_type in ("image", "image_url"):
            parts.append(("image", _media_digest(item.get(item_type, item))))
        elif item_type in ("video", "video_url"):
            parts.append(("video", _media_digest(item.get(item_type, item))))
        elif "text" in item:
            parts.append(("text", item["text"]))
        else:
            parts.append(("opaque", _freeze(item)))
    return tuple(parts)


def _canonicalize_tool_calls(tool_calls: Any) -> tuple | None:
    if not isinstance(tool_calls, list):
        return None
    frozen: list[Any] = []
    for call in tool_calls:
        if not isinstance(call, dict):
            frozen.append(("raw", _freeze(call)))
            continue
        call_id = call.get("id")
        function = call.get("function")
        if isinstance(function, dict):
            name = function.get("name")
            args = canonicalize_tool_arguments(function.get("arguments"))
        else:
            name = None
            args = None
        frozen.append((call_id, name, args))
    return tuple(frozen)


@dataclass(frozen=True)
class MessageKey:
    """Hashable identity of a single chat message, used as a trie child key.

    ``reasoning_content`` is intentionally excluded in M1: think blocks are
    echoed back inconsistently by harnesses/templates and would cause spurious
    forks (TODO: revisit for thinking-model replay). Request-level ``tools`` are
    likewise not part of the key — they are gated separately at prepare time.
    """

    role: str
    content: str | tuple | None
    name: str | None = None
    tool_calls: tuple | None = None
    tool_call_id: str | None = None


def make_message_key(message: dict[str, Any]) -> MessageKey:
    """Build a :class:`MessageKey` from an OpenAI-shaped message dict."""
    return MessageKey(
        role=message.get("role", ""),
        content=canonicalize_content(message.get("content")),
        name=message.get("name"),
        tool_calls=_canonicalize_tool_calls(message.get("tool_calls")),
        tool_call_id=message.get("tool_call_id"),
    )


# ---------------------------------------------------------------------------
# Trie data structures
# ---------------------------------------------------------------------------


def clone_trajectory_buffer(buffer: TrajectoryBuffer) -> TrajectoryBuffer:
    """Deep-copy the mutable list fields so each generation owns its buffer."""
    return TrajectoryBuffer(
        prompt_ids=list(buffer.prompt_ids),
        response_ids=list(buffer.response_ids),
        response_mask=list(buffer.response_mask),
        response_logprobs=list(buffer.response_logprobs),
    )


@dataclass
class BranchCheckpoint:
    """Token-level state captured at a committed assistant prefix.

    Cloned by ``prepare`` so a new request can continue from this point without
    re-encoding the shared prefix. ``image_data``/``video_data`` hold the
    multimodal payload introduced by *this* node's message (per-branch storage;
    decision 4B). The full branch multimodal sequence is rebuilt by walking
    ancestors.
    """

    trajectory_buffer: TrajectoryBuffer
    request_tools: list[dict[str, Any]] | None = None
    chat_template_kwargs_key: tuple | None = None
    messages: list[dict[str, Any]] | None = None
    image_data: list[Any] | None = None
    video_data: list[Any] | None = None
    extra_fields: dict[str, Any] | None = None


@dataclass
class TrieNode:
    """A single message in the session trie.

    Only committed assistant nodes carry a ``checkpoint``. Prompt-side nodes
    materialized during ``prepare`` are structural/pending: ``checkpoint`` is
    ``None`` and they are never exported until an assistant child commits.
    """

    key: MessageKey | None = None
    message: dict[str, Any] | None = None
    parent: TrieNode | None = None
    checkpoint: BranchCheckpoint | None = None
    children: dict[MessageKey, TrieNode] = field(default_factory=dict)
    # Per-node multimodal introduced by this message (mirrors checkpoint copy for
    # prompt-side nodes that do not yet have a checkpoint).
    image_data: list[Any] | None = None
    video_data: list[Any] | None = None
    # In-flight generations attached beneath this node (failure-cleanup hook).
    inflight: int = 0

    @property
    def is_root(self) -> bool:
        return self.parent is None


@dataclass(frozen=True)
class BranchHandle:
    """Opaque token returned by ``prepare`` and passed back to ``commit``.

    The gateway never inspects it; internally it points at the pending attach
    node for this generation.
    """

    generation_id: str


@dataclass
class PrepareResult:
    """Public contract between the trie and the gateway for one generation.

    ``trajectory_buffer`` is a request-local clone of the nearest checkpoint, or
    ``None`` when no checkpoint covers the prefix (full-encode path).
    ``checkpoint_messages`` is the message prefix that buffer already covers, so
    the gateway only encodes ``messages[len(checkpoint_messages):]``.
    """

    trajectory_buffer: TrajectoryBuffer | None
    checkpoint_messages: list[dict[str, Any]]
    branch_handle: BranchHandle
    image_data: list[Any] | None = None
    video_data: list[Any] | None = None
    request_tools: list[dict[str, Any]] | None = None


class PrefixTrie:
    """Per-session prefix trie of chat messages with token checkpoints."""

    def __init__(self) -> None:
        self.root = TrieNode()
        # Maps a live BranchHandle.generation_id to its pending attach node.
        self._pending: dict[str, TrieNode] = {}

    # -- matching -----------------------------------------------------------

    def match(self, messages: list[dict[str, Any]]) -> tuple[TrieNode, int]:
        """Longest-prefix walk from the root.

        Returns the deepest matched node and the number of messages consumed.
        """
        node = self.root
        depth = 0
        for message in messages:
            child = node.children.get(make_message_key(message))
            if child is None:
                break
            node = child
            depth += 1
        return node, depth

    def materialize_prompt_suffix(
        self, node: TrieNode, messages: list[dict[str, Any]], matched_depth: int
    ) -> TrieNode:
        """Find-or-create the prompt-side nodes for ``messages[matched_depth:]``.

        Newly created nodes are structural/pending (no checkpoint). Returns the
        deepest node, which becomes the attach parent for the generation.
        """
        attach = node
        for message in messages[matched_depth:]:
            key = make_message_key(message)
            child = attach.children.get(key)
            if child is None:
                # Shallow-copy so external mutation of the request payload cannot
                # corrupt the stored node.
                child = TrieNode(key=key, message=dict(message), parent=attach)
                attach.children[key] = child
            attach = child
        return attach

    @staticmethod
    def nearest_ckpt(node: TrieNode) -> tuple[TrieNode | None, BranchCheckpoint | None]:
        """Walk up from ``node`` (inclusive) to the nearest node with a checkpoint."""
        current: TrieNode | None = node
        while current is not None:
            if current.checkpoint is not None:
                return current, current.checkpoint
            current = current.parent
        return None, None

    # -- prefix reconstruction ---------------------------------------------

    @staticmethod
    def _path_to_root(node: TrieNode) -> list[TrieNode]:
        chain: list[TrieNode] = []
        current: TrieNode | None = node
        while current is not None and not current.is_root:
            chain.append(current)
            current = current.parent
        chain.reverse()
        return chain

    def rebuild_messages(self, node: TrieNode) -> list[dict[str, Any]]:
        """Rebuild the message transcript from the root to ``node`` (inclusive).

        Returns shallow copies of the stored message dicts so a caller that
        re-keys top-level fields (e.g. during normalization) cannot mutate the
        trie's nodes in place. Shallow (not deep) keeps multimodal payloads from
        being duplicated — nested content is never mutated by current callers.
        """
        return [dict(n.message) for n in self._path_to_root(node) if n.message is not None]

    def collect_multi_modal(self, node: TrieNode) -> tuple[list[Any] | None, list[Any] | None]:
        """Collect per-branch image/video data along the path to ``node``."""
        images: list[Any] = []
        videos: list[Any] = []
        for n in self._path_to_root(node):
            if n.image_data:
                images.extend(n.image_data)
            if n.video_data:
                videos.extend(n.video_data)
        return (images or None, videos or None)

    # -- lifecycle ----------------------------------------------------------

    def prepare(
        self,
        messages: list[dict[str, Any]],
        *,
        generation_id: str | None = None,
    ) -> PrepareResult:
        """Match the incoming messages, materialize pending nodes, and clone the
        nearest checkpoint for a request-local buffer."""
        node, depth = self.match(messages)
        attach_node = self.materialize_prompt_suffix(node, messages, depth)
        ckpt_node, ckpt = self.nearest_ckpt(attach_node)

        generation_id = generation_id or uuid.uuid4().hex
        handle = BranchHandle(generation_id=generation_id)
        self._pending[generation_id] = attach_node
        attach_node.inflight += 1

        if ckpt is None:
            return PrepareResult(
                trajectory_buffer=None,
                checkpoint_messages=[],
                branch_handle=handle,
                image_data=None,
                video_data=None,
            )

        checkpoint_messages = ckpt.messages or self.rebuild_messages(ckpt_node)
        images, videos = self.collect_multi_modal(ckpt_node)
        return PrepareResult(
            trajectory_buffer=clone_trajectory_buffer(ckpt.trajectory_buffer),
            checkpoint_messages=list(checkpoint_messages),
            branch_handle=handle,
            image_data=images,
            video_data=videos,
            request_tools=ckpt.request_tools,
        )

    def upsert_assistant(
        self,
        parent: TrieNode,
        assistant_msg: dict[str, Any],
        checkpoint: BranchCheckpoint,
    ) -> TrieNode:
        """Attach (or refresh) the assistant child under ``parent``.

        Identical assistant output reuses the existing node (idempotent retry);
        differing output creates a new sibling (best-of-N).
        """
        key = make_message_key(assistant_msg)
        child = parent.children.get(key)
        # Shallow-copy so later mutation of the assistant message cannot corrupt
        # the stored node.
        if child is None:
            child = TrieNode(key=key, message=dict(assistant_msg), parent=parent)
            parent.children[key] = child
        else:
            child.message = dict(assistant_msg)
        child.checkpoint = checkpoint
        # Mirror the checkpoint's multimodal payload onto the node, clearing any
        # stale data so a refresh (idempotent retry) stays consistent.
        child.image_data = list(checkpoint.image_data) if checkpoint.image_data is not None else None
        child.video_data = list(checkpoint.video_data) if checkpoint.video_data is not None else None
        return child

    def commit(
        self,
        branch_handle: BranchHandle,
        trajectory_buffer: TrajectoryBuffer,
        assistant_msg: dict[str, Any],
        *,
        request_tools: list[dict[str, Any]] | None = None,
        chat_template_kwargs_key: tuple | None = None,
        messages: list[dict[str, Any]] | None = None,
        image_data: list[Any] | None = None,
        video_data: list[Any] | None = None,
        extra_fields: dict[str, Any] | None = None,
    ) -> TrieNode:
        """Resolve the pending node and attach the generated assistant child."""
        attach_node = self._pending.pop(branch_handle.generation_id, None)
        if attach_node is None:
            raise KeyError(f"Unknown or already-committed branch_handle: {branch_handle.generation_id}")
        attach_node.inflight = max(0, attach_node.inflight - 1)
        # The checkpoint lives on the assistant node and its buffer covers the
        # request prompt PLUS the generated assistant turn, so the stored prefix
        # must include ``assistant_msg`` (this is what the next turn's
        # ``checkpoint_messages`` slices against).
        covered_messages = [dict(m) for m in messages] + [dict(assistant_msg)] if messages is not None else None
        checkpoint = BranchCheckpoint(
            trajectory_buffer=trajectory_buffer,
            request_tools=request_tools,
            chat_template_kwargs_key=chat_template_kwargs_key,
            messages=covered_messages,
            image_data=image_data,
            video_data=video_data,
            extra_fields=extra_fields,
        )
        return self.upsert_assistant(attach_node, assistant_msg, checkpoint)

    def abandon(self, branch_handle: BranchHandle) -> None:
        """Release a pending generation that failed before commit.

        Decrements the attach node's in-flight count. A childless structural
        node is left in place (harmless; skipped by export) — M1 relies on the
        finalize sweep rather than eager detach (TODO: refcount detach).
        """
        attach_node = self._pending.pop(branch_handle.generation_id, None)
        if attach_node is not None:
            attach_node.inflight = max(0, attach_node.inflight - 1)

    # -- export -------------------------------------------------------------

    def iter_export_nodes(self, *, export_all: bool = False) -> Iterator[TrieNode]:
        """Yield assistant checkpoint nodes for finalize.

        By default emits only *terminal* assistant checkpoints (no committed
        assistant descendant): rejected best-of-N siblings stay short leaves and
        a continued branch is represented by its deepest checkpoint. With
        ``export_all`` every committed checkpoint is emitted.

        A single post-order pass (O(N) over the trie) collects the terminals;
        each node reports whether its subtree already carries a checkpoint so an
        ancestor knows whether it is the deepest one.
        """
        if export_all:
            yield from self._iter_checkpoint_nodes(self.root)
            return
        _, terminals = self._collect_terminals(self.root)
        yield from terminals

    def _collect_terminals(self, node: TrieNode) -> tuple[bool, list[TrieNode]]:
        subtree_has_ckpt = False
        terminals: list[TrieNode] = []
        for child in node.children.values():
            child_has_ckpt, child_terminals = self._collect_terminals(child)
            subtree_has_ckpt = subtree_has_ckpt or child_has_ckpt
            terminals.extend(child_terminals)
        if node.checkpoint is not None:
            if not subtree_has_ckpt:
                terminals.append(node)
            subtree_has_ckpt = True
        return subtree_has_ckpt, terminals

    def _iter_checkpoint_nodes(self, node: TrieNode) -> Iterator[TrieNode]:
        for child in node.children.values():
            if child.checkpoint is not None:
                yield child
            yield from self._iter_checkpoint_nodes(child)

    # -- introspection ------------------------------------------------------

    def num_branches(self) -> int:
        """Number of terminal checkpoint leaves (distinct exportable branches)."""
        return sum(1 for _ in self.iter_export_nodes())

    def num_inflight(self) -> int:
        return len(self._pending)
