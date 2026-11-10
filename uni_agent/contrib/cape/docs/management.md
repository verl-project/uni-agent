# Website management API (version 2)

The website uses the same persistent IdentityStore as the native CLI. Initialize
and start the controller using `controller init/start --state-dir` as described in
[identity.md](identity.md). There is no `--management-dir`, second authority DB,
Join Token, or legacy auth-file mode.

`GET /v1/management/capabilities` advertises `management_api: 2` and the
`persistent-groups` identity model. `POST /v1/management/{action}` accepts a JSON
object and an authenticated Bearer credential. Version 1 clients must negotiate
this change rather than silently mapping old per-user grants to groups.

Actions: `me`, `resources`, `pools.list/create/edit/drain/remove`,
`tokens.list/create/revoke`, `users.list/update`, `groups.list/set/delete`,
`nodes.list/enroll/action`, and `audit.list`. Pool/user/group mutations require
an admin credential; a regular credential owned by an admin remains regular.
Node actions use main's owner checks, lifecycle reconciliation and durable journal.
Resources are authenticated public inventory with private execution/session data
removed. `nodes.list` returns only owned nodes unless the caller is admin.

Personal token creation requires an `operation_id` and native token name. It
returns the secret once; replay returns metadata with `already_issued: true`.
Secrets are never recoverable from state or audit. Tokens use the native `cap1_`
format and work through the CLI immediately. Revocation uses the same live store.

Old joins, contribution approval, enrollment reset and per-user pool grant actions
are intentionally unsupported. Use main's owned node enrollment (client saves
its secret before sending the verifier), node controls, and live group membership.
The deployed website v1 must be migrated before connecting it to this API.

## Company SSO

The scheduler never accepts an unverified browser email/name as identity. An admin
issues a narrowly scoped `sso.bridge.create` credential with canonical `tenant_id`,
`client_id`, explicit `auto_provision` and an `operation_id`. Only the trusted
website backend holds that secret. It calls `session.sso` after verifying the
company ID token, passing the configured tenant/client and immutable `subject`.

The resulting one-hour session is linked to the bridge and account, independent
of personal tokens. `session.revoke` signs out that session. Admin-only
`sso.bridge.revoke` takes `bridge_id` and immediately invalidates its sessions.
Disabled accounts, expired sessions and revoked bridges fail authentication.
Bridge credentials cannot call native human/resource/node APIs or register their
verifier as another credential type.

Automatic provisioning creates a regular user with **empty groups and no personal
token**; it deliberately does not inherit main's permissive `default` group.
Admins can pre-provision with `sso.users.create`, or explicitly bind an existing
user using `sso.users.bind` plus `user_id`. Binding requires admin authority and
cannot replace an existing SSO identity or merge accounts by name/email. An
explicitly bound admin account can administer through its SSO session; a regular
token issued from that session still has regular privileges.

SSO state lives in `web_auth` inside the same `authority` snapshot and audit
transaction, not in another SQLite database. First SSO provisioning/bridge issuance
upgrades the authority to version **3**. Version-2 data can be read without an
implicit upgrade; older binaries reject version 3 rather than silently ignoring
session restrictions. Python and Rust can reopen the same version-3 instance.
Do not downgrade authority state or run old binaries on an SSO-enabled instance.

Regular tokens remain native credentials. Their revocation prevents new calls,
not browser sign-in or already accepted work; dispatch retains main's live
account/group checks. Revoking a bridge ends browser sessions without revoking
independently created regular tokens. Secret-producing retries return only
metadata (`already_issued`); callers must initiate a fresh exchange if the original
secret response was lost.
