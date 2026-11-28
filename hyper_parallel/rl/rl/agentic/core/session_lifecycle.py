# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Bounded-memory tombstones shared by external agent gateways."""

import sqlite3


class ReleasedSessions:
    """Retain late-request tombstones on disk with a fixed SQLite page cache."""

    def __init__(self) -> None:
        """Create an automatically deleted temporary database."""
        self.database = sqlite3.connect("", check_same_thread=False, isolation_level=None)
        self.database.execute("PRAGMA cache_size=-64")
        self.database.execute("PRAGMA journal_mode=OFF")
        self.database.execute("PRAGMA synchronous=OFF")
        self.database.execute("CREATE TABLE released (id TEXT PRIMARY KEY) WITHOUT ROWID")

    def add(self, session_id: str) -> None:
        """Remember one released ID without keeping its Python string alive."""
        self.database.execute("INSERT OR IGNORE INTO released VALUES (?)", (session_id,))

    def __contains__(self, session_id: str) -> bool:
        """Reject delayed registration throughout the gateway lifetime."""
        return self.database.execute("SELECT 1 FROM released WHERE id=?", (session_id,)).fetchone() is not None

    def close(self) -> None:
        """Release the page cache and automatically remove the temporary file."""
        self.database.close()
