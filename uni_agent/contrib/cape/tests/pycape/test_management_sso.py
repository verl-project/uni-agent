# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import copy
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
import uuid

from pycape.controller import CAPEController
from pycape.identity import IdentityStore, new_secret, verifier
from pycape.managed_auth import ManagedAuthority
from pycape.management import call, capabilities

TENANT = '11111111-1111-4111-8111-111111111111'
CLIENT = '22222222-2222-4222-8222-222222222222'
SUBJECT = '33333333-3333-4333-8333-333333333333'


class ManagementSSOTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.time = 1000.
        self.path = Path(self.tmp.name) / 'state.sqlite3'
        self.admin = new_secret()
        self.store = IdentityStore.create(self.path, 'admin', self.admin, clock=lambda:self.time)
        self.addCleanup(self.store.close)
        self.controller = CAPEController(auth=ManagedAuthority(self.store), clock=lambda:self.time)
        self.bridge = self.call('sso.bridge.create', self.admin, tenant_id=TENANT, client_id=CLIENT, auto_provision=True)

    def call(self, action, token, **body):
        body.setdefault('operation_id', str(uuid.uuid4()))
        return call(self.controller, action, token, body)

    def login(self, subject=SUBJECT, bridge=None):
        return self.call('session.sso', (bridge or self.bridge)['token'], tenant_id=TENANT, client_id=CLIENT, subject=subject)

    def test_shared_native_identity_and_revocation(self):
        self.assertEqual(capabilities(self.controller)['management_api'], 2)
        login = self.login(); session = login['session_token']; uid = login['user']['id']
        self.assertEqual(self.store.snapshot()['version'], 3)
        self.assertEqual(login['user']['groups'], [])
        self.assertEqual(self.call('tokens.list', session)['tokens'], [])
        pool = self.store.create_pool(self.admin,'gpu')
        self.assertFalse(self.store.can_submit(session,pool))
        self.store.set_group(self.admin,'team',pools=[pool])
        self.store.edit_user(self.admin,uid,groups=['team'])
        self.assertTrue(self.store.can_submit(session,pool))
        issued = self.call('tokens.create',session,name='cli')
        self.assertEqual(self.store.authenticate(issued['token'])['user_id'],uid)
        self.assertTrue(self.store.can_submit(issued['token'],pool))
        self.assertTrue(self.store.pending_allowed(uid,pool,login['id']))
        self.call('tokens.revoke',session,token_id=issued['id'])
        with self.assertRaises(PermissionError):self.store.authenticate(issued['token'])
        self.assertEqual(self.call('me',session)['credential']['auth_method'],'sso')
        self.call('session.revoke',session)
        with self.assertRaises(PermissionError):self.store.authenticate(session)
        serialized = json.dumps(self.store.snapshot())
        for secret in (self.admin,self.bridge['token'],session,issued['token']):self.assertNotIn(secret,serialized)
        self.assertFalse((Path(self.tmp.name)/'management.sqlite').exists())

    def test_bridge_scope_revocation_and_nonhuman_denial(self):
        for action in ('me','resources','tokens.create','nodes.list'):
            with self.assertRaises(PermissionError):self.call(action,self.bridge['token'],name='cli')
        with self.assertRaises(PermissionError):self.call('session.sso',self.bridge['token'],tenant_id=TENANT,client_id=SUBJECT,subject=SUBJECT)
        session=self.login()['session_token']
        for sensitive in (self.bridge['token'],session):
            with self.assertRaises(ValueError):
                self.store.register_token(str(uuid.uuid4()),verifier(sensitive),'stolen',username='duplicate')
        self.call('sso.bridge.revoke',self.admin,bridge_id=self.bridge['id'])
        with self.assertRaises(PermissionError):self.store.authenticate(session)
        with self.assertRaises(PermissionError):self.login()

    def test_disabled_expired_and_manual_provisioning(self):
        bridge=self.call('sso.bridge.create',self.admin,tenant_id=TENANT,client_id=CLIENT,auto_provision=False)
        with self.assertRaises(PermissionError):self.login(bridge=bridge)
        user=self.call('sso.users.create',self.admin,tenant_id=TENANT,subject=SUBJECT)
        session=self.login(bridge=bridge)['session_token']
        self.store.edit_user(self.admin,user['id'],disabled=True)
        with self.assertRaises(PermissionError):self.store.authenticate(session)
        with self.assertRaises(PermissionError):self.login(bridge=bridge)
        self.store.edit_user(self.admin,user['id'],disabled=False)
        self.time += 3601
        with self.assertRaises(PermissionError):self.store.authenticate(session)

    def test_admin_binding_does_not_elevate_regular_token(self):
        uid=self.store.authenticate(self.admin)['user_id']
        self.call('sso.users.bind',self.admin,tenant_id=TENANT,subject=SUBJECT,user_id=uid)
        login=self.login(); self.assertTrue(login['user']['platform_admin'])
        issued=self.call('tokens.create',login['session_token'],name='cli')
        self.assertFalse(self.store.authenticate(issued['token'])['is_admin'])
        with self.assertRaises(PermissionError):self.call('groups.set',issued['token'],name='bad')
        with self.assertRaises(ValueError):self.call('sso.users.bind',self.admin,tenant_id=TENANT,subject=CLIENT,user_id=uid)

    def test_metadata_only_replay_and_no_foreign_token_access(self):
        a=self.login(); b=self.login(subject=CLIENT)
        op=str(uuid.uuid4()); first=self.call('tokens.create',a['session_token'],operation_id=op,name='cli')
        replay=self.call('tokens.create',a['session_token'],operation_id=op,name='cli')
        self.assertEqual(first['id'],replay['id']);self.assertNotIn('token',replay)
        with self.assertRaises(ValueError):self.call('tokens.create',a['session_token'],operation_id=op,name='other')
        with self.assertRaises(PermissionError):self.call('tokens.revoke',b['session_token'],token_id=first['id'])
        self.assertEqual(self.call('tokens.list',b['session_token'])['tokens'],[])
        with self.assertRaises(PermissionError):self.call('users.update',a['session_token'],user_id=b['user']['id'],groups=['default'])
        # Accepted work still checks current groups, independent of admission expiry.
        pool=self.store.create_pool(self.admin,'gpu');uid=a['user']['id']
        self.store.edit_user(self.admin,uid,groups=['default'])
        self.assertTrue(self.store.pending_allowed(uid,pool,a['id']))
        self.store.edit_user(self.admin,uid,groups=[])
        self.assertFalse(self.store.pending_allowed(uid,pool,a['id']))

    def test_unknown_sso_parameters_are_not_persisted(self):
        before=self.store.snapshot()
        with self.assertRaises(ValueError):
            self.call('session.sso',self.bridge['token'],tenant_id=TENANT,client_id=CLIENT,subject=SUBJECT,unexpected_secret='do-not-persist')
        self.assertEqual(before,self.store.snapshot())

    def test_schema_fence_and_invalid_extension(self):
        self.assertEqual(self.store.snapshot()['version'],3)
        original = self.store.snapshot()
        for bad in ({}, {'bridges':[], 'sessions':{}}):
            state=copy.deepcopy(original);state['web_auth']=bad
            self.store.db.execute('UPDATE authority SET body=?',(json.dumps(state),))
            with self.assertRaises(ValueError):self.store.snapshot()
            self.store.db.execute('UPDATE authority SET body=?',(json.dumps(original),))
