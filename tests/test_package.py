"""Smoke tests for the root Python package layout."""

from trigger_agent_misc import __version__


def test_version_is_semantic_string() -> None:
    parts = __version__.split(".")
    assert len(parts) == 3
    assert all(part.isdigit() for part in parts)
