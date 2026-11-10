# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import socket
import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from tests.managed_fixture import ManagedFixture, command


PYCAPE_BIN = os.environ.get("PYCAPE_BIN", "pycape")
RSCAPE_BIN = os.environ.get("RSCAPE_BIN", "rscape")


class AuthenticationInteropTests(unittest.TestCase):
    def command(self, binary, *args):
        return subprocess.run([binary, *args], text=True, capture_output=True)

    def scenario(self, writer, controller_binary, client):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = ManagedFixture(writer, tmp)
            controller = fixture.start(controller_binary)
            url = fixture.url
            try:
                fixture.pool("pool-a")
                alice = fixture.user_file("alice")
                bob = fixture.user_file("bob")
                tokens = [fixture.user_token("alice"), fixture.user_token("bob")]
                database = (fixture.state / "state.sqlite3").read_bytes()
                for secret in tokens:
                    self.assertNotIn(secret.encode(), database)
                submitted=self.command(client,"run","--controller-url",url,"--token-file",str(alice),"--pool","pool-a","--user","alice","--session-id","auth","--","true")
                self.assertEqual(submitted.returncode,0,submitted.stderr)
                request_id=submitted.stdout.strip()
                own=self.command(client,"status",request_id,"--controller-url",url,"--token-file",str(alice))
                self.assertEqual(own.returncode,0,own.stderr)
                denied=self.command(client,"status",request_id,"--controller-url",url,"--token-file",str(bob))
                self.assertNotEqual(denied.returncode,0)
            finally:
                controller.terminate(); controller.communicate(timeout=5)

    def test_pycape_identity_is_read_by_rscape(self):
        self.scenario(PYCAPE_BIN,RSCAPE_BIN,RSCAPE_BIN)

    def test_rscape_identity_is_read_by_pycape(self):
        self.scenario(RSCAPE_BIN,PYCAPE_BIN,PYCAPE_BIN)


if __name__ == "__main__": unittest.main()
