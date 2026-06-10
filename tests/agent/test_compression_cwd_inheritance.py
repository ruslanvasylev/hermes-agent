"""Regression: compression continuation must inherit the parent session's cwd.

When a session with an explicit workspace (cwd) is compressed, the continuation
session should preserve that cwd. Otherwise list_sessions_rich() projects the
tip's NULL cwd over the root's valid cwd, and the conversation disappears into
the "No workspace" group in Desktop/TUI.

See: https://github.com/NousResearch/hermes-agent/issues/42228
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from hermes_state import SessionDB


def _build_agent_with_db(db: SessionDB, session_id: str, cwd: str = None):
    """Build an AIAgent wired to db with a specific session cwd."""
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            quiet_mode=True,
            session_db=db,
            session_id=session_id,
            skip_context_files=True,
            skip_memory=True,
        )

    # Create the initial session with a cwd
    db.create_session(
        session_id=session_id,
        source="cli",
        model="test/model",
        cwd=cwd,
    )

    # Stub compressor to return deterministic output without LLM calls
    compressor = MagicMock()

    def _compress(*_a, **_kw):
        return [
            {"role": "user", "content": "[CONTEXT COMPACTION] summary"},
            {"role": "user", "content": "tail"},
        ]

    compressor.compress.side_effect = _compress
    compressor.compression_count = 1
    compressor.last_prompt_tokens = 0
    compressor.last_completion_tokens = 0
    compressor._last_summary_error = None
    compressor._last_compress_aborted = False
    compressor._last_aux_model_failure_model = None
    compressor._last_aux_model_failure_error = None
    agent.context_compressor = compressor
    return agent


class TestCompressionCwdInheritance:
    """Continuation sessions must inherit the parent's cwd."""

    def test_continuation_inherits_cwd(self, tmp_path: Path):
        """After compression, the new session should have the same cwd as the old one."""
        db_path = tmp_path / "test.db"
        db = SessionDB(db_path)

        session_id = "20260608_120000_abc123"
        workspace = "/Users/test/my-project"
        agent = _build_agent_with_db(db, session_id, cwd=workspace)

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello " * 2000},
            {"role": "assistant", "content": "Hi there!"},
        ]

        from agent.conversation_compression import compress_context

        with patch.object(agent, "_build_system_prompt", return_value="system"):
            compress_context(agent, messages, "You are helpful.", force=True)

        # The session_id should have changed (new continuation)
        assert agent.session_id != session_id

        # The new session should have inherited the cwd
        new_session = db.get_session(agent.session_id)
        assert new_session is not None
        assert new_session["cwd"] == workspace

    def test_continuation_inherits_null_cwd(self, tmp_path: Path):
        """Sessions without a workspace should keep cwd=None after compression."""
        db_path = tmp_path / "test.db"
        db = SessionDB(db_path)

        session_id = "20260608_120000_def456"
        agent = _build_agent_with_db(db, session_id, cwd=None)

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello " * 2000},
            {"role": "assistant", "content": "Hi there!"},
        ]

        from agent.conversation_compression import compress_context

        with patch.object(agent, "_build_system_prompt", return_value="system"):
            compress_context(agent, messages, "You are helpful.", force=True)

        new_session = db.get_session(agent.session_id)
        assert new_session is not None
        assert new_session["cwd"] is None

    def test_tip_projection_preserves_cwd(self, tmp_path: Path):
        """list_sessions_rich tip projection should show the inherited cwd."""
        db_path = tmp_path / "test.db"
        db = SessionDB(db_path)

        root_id = "20260608_120000_root01"
        workspace = "/Users/test/my-project"
        agent = _build_agent_with_db(db, root_id, cwd=workspace)

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello " * 2000},
            {"role": "assistant", "content": "Hi there!"},
        ]

        from agent.conversation_compression import compress_context

        with patch.object(agent, "_build_system_prompt", return_value="system"):
            compress_context(agent, messages, "You are helpful.", force=True)

        tip_id = agent.session_id

        # list_sessions_rich should project the tip's cwd onto the root
        sessions = db.list_sessions_rich(source="cli", project_compression_tips=True)
        # Find the projected session (root projected to tip)
        projected = [s for s in sessions if s.get("_lineage_root_id") == root_id]
        assert len(projected) == 1
        assert projected[0]["cwd"] == workspace
        assert projected[0]["id"] == tip_id
