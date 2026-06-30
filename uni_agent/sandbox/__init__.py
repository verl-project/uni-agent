"""Sandbox providers -- one module per provider, one class per provider.

Each provider is a single :class:`Sandbox` subclass that owns its lifecycle and
*is* the data plane (exec + file transfer + optional port tunnel). Tools and the
channels they hold depend only on the narrow :class:`SandboxBackend` protocol
(the data-plane subset), never on lifecycle.

``base`` and host-local providers depend only on the standard library and are
always imported. Provider modules that need a third-party SDK are imported
*defensively*: importing this package must never force every provider's
dependency to be installed, so a provider whose SDK is missing is skipped and
its symbol is left as ``None`` (and kept out of ``__all__``).

Add a new provider by dropping a ``<name>.py`` next to this file and, if it has
a heavy/optional dependency, registering it in the ``try/except`` block below.
"""

from __future__ import annotations

from .base import ExecResult, Sandbox, SandboxBackend
from .local import LocalSandbox

__all__ = ["ExecResult", "SandboxBackend", "Sandbox", "LocalSandbox"]

# ----- optional providers: skip cleanly when their SDK isn't installed -----
# (``modal`` is imported lazily inside ``ModalSandbox.start`` and referenced
# under TYPE_CHECKING, so this import only fails if the module ever grows a hard
# provider import -- the guard keeps the package importable regardless.)
try:
    from .modal import ModalSandbox
except ImportError:
    ModalSandbox = None  # type: ignore[assignment, misc]
else:
    __all__.append("ModalSandbox")
