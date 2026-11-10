# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real HTTP management v2 and SSO across both native authority implementations."""
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import tempfile
import time
import unittest
import uuid
from pycape.cli import _request, CapeHTTPError
from pycape.credential_file import read_registration
from pycape.identity import IdentityStore, new_secret, verifier

TENANT='11111111-1111-4111-8111-111111111111'
CLIENT='22222222-2222-4222-8222-222222222222'
SUBJECT='33333333-3333-4333-8333-333333333333'

@unittest.skipUnless(os.environ.get('CAPE_MANAGEMENT_TEST')=='1','enable native HTTP parity')
class SSOParityTests(unittest.TestCase):
    def run_cli(self,binary,*args):
        result=subprocess.run([binary,*args],text=True,capture_output=True,timeout=20)
        self.assertEqual(result.returncode,0,result.stderr)
        return json.loads(result.stdout)

    def start(self,binary,directory):
        with socket.socket() as sock:
            sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
        process=subprocess.Popen([binary,'controller','start','--state-dir',str(directory/'state'),
            '--token-file',str(directory/'admin.json'),'--port',str(port)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        url='http://127.0.0.1:'+str(port)
        try:
            for _ in range(200):
                if process.poll() is not None:self.fail('controller exited: '+process.communicate()[1])
                try:
                    _request('GET',url,'/health',timeout=1);return process,url
                except Exception:time.sleep(.05)
            self.fail('controller startup timeout')
        except BaseException:
            self.stop(process);raise

    def stop(self,process):
        if process.poll() is None:process.send_signal(signal.SIGINT)
        try:process.communicate(timeout=15)
        except subprocess.TimeoutExpired:process.kill();process.communicate();self.fail('controller did not stop')

    def denied(self,fn,status=403):
        with self.assertRaises(CapeHTTPError) as raised:fn()
        self.assertEqual(raised.exception.status,status)

    def test_native_sso_revocation_and_bidirectional_store_reopen(self):
        python=os.environ.get('PYCAPE_BIN','pycape');rust=os.environ.get('RSCAPE_BIN','rscape')
        for first,second in ((python,rust),(rust,python)):
            with self.subTest(first=first),tempfile.TemporaryDirectory() as tmp:
                directory=Path(tmp)
                self.run_cli(first,'controller','init','--state-dir',str(directory/'state'),'--user','admin','--token-out',str(directory/'admin.json'))
                admin=read_registration(directory/'admin.json')['secret']
                process,url=self.start(first,directory)
                def call(endpoint,token=admin,**body):
                    body.setdefault('operation_id',str(uuid.uuid4()))
                    return _request('POST',url,'/v1/management/'+endpoint,payload=body,token=token)
                try:
                    caps=_request('GET',url,'/v1/management/capabilities');self.assertEqual(caps['management_api'],2)
                    self.denied(lambda:call('me',None))
                    bridge=call('sso.bridge.create',tenant_id=TENANT,client_id=CLIENT,auto_provision=True)
                    for action in ('me','resources','users.list','groups.list'):
                        self.denied(lambda:call(action,bridge['token']))
                    self.denied(lambda:call('session.sso',bridge['token'],tenant_id=TENANT,client_id=SUBJECT,subject=SUBJECT))
                    login=call('session.sso',bridge['token'],tenant_id=TENANT,client_id=CLIENT,subject=SUBJECT)
                    session=login['session_token'];uid=login['user']['id']
                    self.assertEqual(login['user']['groups'],[])
                    for sensitive in (bridge['token'],session):
                        self.denied(lambda:_request('POST',url,'/v1/identity/tokens',payload={'operation_id':str(uuid.uuid4()),'verifier':verifier(sensitive),'name':'stolen','username':'duplicate'}),400)
                    self.assertEqual(call('tokens.list',session)['tokens'],[])
                    self.denied(lambda:call('groups.set',session,name='escalate',all_pools=True))
                    pool=call('pools.create',name='gpu')['id']
                    call('groups.set',name='team',pools=[pool]);call('users.update',user_id=uid,groups=['team'])
                    self.assertEqual(call('me',session)['user']['grants'],{pool:['submit']})
                    op=str(uuid.uuid4());issued=call('tokens.create',session,name='cli',operation_id=op)
                    again=call('tokens.create',session,name='cli',operation_id=op)
                    self.assertEqual(again['id'],issued['id']);self.assertNotIn('token',again)
                    actor=_request('GET',url,'/v1/identity/me',token=issued['token'])
                    self.assertEqual(actor['user_id'],uid);self.assertFalse(actor['is_admin'])
                    # User-issued token belongs to the native authority, including admission.
                    workload={'user_id':uid,'pool':pool,'session_id':'sso-parity','command':['true'],'resources':{'gpu_count':0,'cpu_cores':1,'memory_gb':1}}
                    submitted=_request('POST',url,'/v1/submit',payload=workload,token=issued['token'])
                    self.assertIn('request_id',submitted)
                    other=call('session.sso',bridge['token'],tenant_id=TENANT,client_id=CLIENT,subject=CLIENT)['session_token']
                    node = {'operation_id':str(uuid.uuid4()), 'node_id':'web-node', 'pool':pool,
                            'allocation_id':'web-allocation', 'agent_epoch':'web-epoch',
                            'resources':{'cpu_cores':1,'memory_gb':1,'gpu_uuids':[]}}
                    for sensitive in (bridge['token'],session):
                        self.denied(lambda:call('nodes.enroll',session,**dict(node,verifier=verifier(sensitive))),400)
                    enrolled=call('nodes.enroll',session,**dict(node,verifier=verifier(new_secret())))
                    self.assertEqual(enrolled['node_id'],'web-node')
                    self.assertEqual(len(call('nodes.list',session)['nodes']),1)
                    self.assertEqual(call('nodes.list',other)['nodes'],[])
                    self.denied(lambda:call('nodes.action',other,node_id='web-node',action='drain'))
                    call('nodes.action',session,node_id='web-node',action='drain')
                    self.denied(lambda:call('tokens.revoke',other,token_id=issued['id']))
                    self.denied(lambda:call('session.sso',issued['token'],tenant_id=TENANT,client_id=CLIENT,subject=SUBJECT))
                    call('tokens.revoke',session,token_id=issued['id'])
                    self.denied(lambda:_request('GET',url,'/v1/identity/me',token=issued['token']))
                    self.assertEqual(call('me',session)['credential']['auth_method'],'sso')
                    # Same credential type rule for an explicitly bound admin account.
                    admin_uid=call('me')['user']['id']
                    call('sso.users.bind',tenant_id=TENANT,subject=TENANT,user_id=admin_uid)
                    admin_session=call('session.sso',bridge['token'],tenant_id=TENANT,client_id=CLIENT,subject=TENANT)['session_token']
                    self.assertTrue(call('me',admin_session)['user']['platform_admin'])
                    regular=call('tokens.create',admin_session,name='personal')['token']
                    self.denied(lambda:call('users.list',regular))
                    call('session.revoke',other)
                    self.denied(lambda:call('me',other))
                    self.assertEqual(call('resources',session)['pools'][0]['id'],pool)
                    retained=call('tokens.create',session,name='retained')['token']
                    self.denied(lambda:call('joins.create',pool=pool),400)
                finally:self.stop(process)
                store=IdentityStore(directory/'state/state.sqlite3')
                try:
                    snapshot=store.snapshot();self.assertEqual(snapshot['version'],3)
                    raw=json.dumps(snapshot)
                    for secret in (admin,bridge['token'],session,issued['token'],retained):self.assertNotIn(secret,raw)
                    self.assertFalse(store.authenticate(retained)['is_admin'])
                finally:store.close()
                process,url=self.start(second,directory)
                try:
                    self.assertEqual(call('me',session)['user']['id'],uid)
                    self.assertEqual(_request('GET',url,'/v1/identity/me',token=retained)['user_id'],uid)
                    call('sso.bridge.revoke',bridge_id=bridge['id'])
                    self.denied(lambda:call('me',session))
                    # Bridge revocation does not revoke independently created native tokens.
                    self.assertEqual(_request('GET',url,'/v1/identity/me',token=retained)['user_id'],uid)
                finally:self.stop(process)
