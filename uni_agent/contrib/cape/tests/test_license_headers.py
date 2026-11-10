# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import os
import re
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]
APACHE_COPYRIGHT = (
    "# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved."
)
RUST_COPYRIGHT = APACHE_COPYRIGHT.replace("# ", "// ", 1)
CUTLASS_PATH = Path("tests/mimic_use_case/cutedsl_add.py")
CUTLASS_HEADER = """# Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-""" + """Identifier: BSD-3-Clause

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE."""
PRIVATE_PATTERN = re.compile(
    rb"(?i)(/home/" + rb"scratch\.|scratch\.[a-z]+_sw|\b" + rb"jian" + rb"bingd\b|"
    rb"/local_" + rb"nvme/|\.nvidia\.com|gitlab-" + rb"master|nv" + rb"bugs?|"
    rb"[a-z-]+-slurm-[0-9]+)"
)


def tracked_files():
    try:
        output = subprocess.check_output(["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=ROOT)
        return [ROOT / item.decode() for item in output.split(b"\0") if item]
    except (OSError, subprocess.CalledProcessError):
        excluded = {".git", "target", "__pycache__", "build"}
        files = []
        for base, directories, names in os.walk(ROOT):
            directories[:] = [name for name in directories if name not in excluded]
            files.extend(Path(base) / name for name in names)
        return files


class LicenseHeaderTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.files = tracked_files()

    def test_source_headers_and_shebangs(self):
        for path in self.files:
            relative = path.relative_to(ROOT)
            if relative == CUTLASS_PATH:
                continue
            if not (path.suffix in {".py", ".rs", ".sh", ".toml"} or path.name == "Makefile"):
                continue
            data = path.read_bytes()
            head = data.splitlines()[:20]
            identifier = b"SPDX-License-" + b"Identifier: Apache-2.0"
            self.assertEqual(sum(identifier in line for line in head), 1, relative)
            expected = RUST_COPYRIGHT if path.suffix == ".rs" else APACHE_COPYRIGHT
            self.assertIn(expected.encode(), head, relative)
            if data.startswith(b"#!"):
                self.assertTrue(data.splitlines()[0].startswith(b"#!"), relative)

    def test_cutlass_header_is_exact_and_exclusive(self):
        data = (ROOT / CUTLASS_PATH).read_text()
        self.assertEqual("\n".join(data.splitlines()[:27]), CUTLASS_HEADER)
        identifier = "SPDX-License-" + "Identifier: "
        self.assertEqual(data.count(identifier + "BSD-3-Clause"), 1)
        self.assertNotIn(identifier + "Apache-2.0", data)

    def test_public_tree_has_no_private_paths_or_closed_license_markers(self):
        for path in self.files:
            data = path.read_bytes()
            if b"\0" in data:
                continue
            relative = path.relative_to(ROOT)
            # Public CUDA package distribution is not an internal service hostname.
            data = data.replace(b"https://developer.download.nvidia.com/compute/cuda/repos/", b"PUBLIC_CUDA_REPOSITORY/")
            self.assertIsNone(PRIVATE_PATTERN.search(data), relative)
            if path.name in {"Cargo.toml", "pyproject.toml"}:
                self.assertNotRegex(data.decode(), "(?i)proprie" + "tary")

    def test_repository_notices_exist(self):
        self.assertTrue((ROOT / "LICENSE").is_file())
        notices = (ROOT / "THIRD-PARTY-NOTICES.txt").read_text()
        self.assertIn("CUTLASS", notices)
        self.assertIn("SQLite", notices)


if __name__ == "__main__":
    unittest.main()
