# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Owner-authorized logical-session queries and execution-state operations."""

from .models import SessionState, RequestState, CommandState, REQUEST_TERMINAL_STATES


def list_sessions(controller, token, session_id=None):
    if not getattr(controller.auth, "managed", False):
        raise PermissionError("persistent identity is required")
    actor = controller.auth.store.authenticate(token)
    owner = actor["user_id"]
    result = []
    for (user, identity), logical in controller.logical_sessions.items():
        if user != owner or (session_id is not None and identity != session_id):
            continue
        item = dict(logical)
        item["executions"] = [controller._session_details(s) for s in controller.sessions.values()
                              if s.user_id == user and s.session_id == identity]
        result.append(item)
    if session_id is not None and not result:
        raise ValueError("logical session not found")
    return {"sessions": result}


def execution_action(controller, execution_id, action, token, force=False):
    if action not in ("suspend", "purge") or (action == "purge" and force):
        raise ValueError("invalid execution action")
    session = controller.sessions[execution_id]
    controller._authorize_session_operator(session, token)
    if action == "suspend":
        if session.state in (SessionState.SUSPENDING, SessionState.SUSPENDED):
            return controller._session_details(session)
        if session.state == SessionState.RUNNING_COMMAND:
            session.suspend_requested = True
            if force:
                history = [c for c in controller.command_assignments.values() if c.execution_session_id == execution_id]
                never_dispatched = bool(history) and all(c.state == CommandState.PENDING or
                    (c.state == CommandState.CANCELLED and c.result.get("was_dispatched") is False) for c in history)
                request = controller.requests[session.active_request_id]
                controller._mark_request_terminal(request, RequestState.CANCELLED)
                request.reason = "forced_suspend"
                command = controller.request_commands.get(request.request_id)
                if command:
                    controller.cancel_command(command, reason="forced_suspend")
                controller._emit("request_state", request.request_id, request.state.value, request.reason,
                                 controller._request_details(request))
                session.active_request_id = None
                session.state = SessionState.CACHED_IDLE
                if never_dispatched:
                    controller._suspend_unstarted(session)
                    return controller._session_details(session)
            else:
                controller._emit("session_state", execution_id, session.state.value, "suspend_requested",
                                 controller._session_details(session))
                return controller._session_details(session)
        controller.suspend_session(execution_id, token=token)
    else:
        if session.state == SessionState.PURGED:
            return controller._session_details(session)
        if session.state not in (SessionState.SUSPENDED, SessionState.PURGING):
            raise ValueError("purge requires a suspended execution")
        if any(r.state not in REQUEST_TERMINAL_STATES and session.matches_reuse_identity(
                r.spec, r.spec.image_identity(controller.policy.default_image), session.runtime_adapter)
               for r in controller.requests.values()):
            raise ValueError("execution is referenced by pending requests")
        if session.empty_retained:
            session.state = SessionState.PURGED
            controller._emit("session_state", execution_id, session.state.value, "purged", controller._session_details(session))
            return controller._session_details(session)
        session.state = SessionState.PURGING
        controller._emit("session_state", execution_id, session.state.value, "purge_requested",
                         controller._session_details(session))
        command = {"action": "purge_session", "execution_session_id": execution_id,
                   "suspend_version": session.suspend_version}
        queue = controller.node_control_queues.setdefault(session.node_id, [])
        if command not in queue:
            queue.append(command)
    return controller._session_details(session)
