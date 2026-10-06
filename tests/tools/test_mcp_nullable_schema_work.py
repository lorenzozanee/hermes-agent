"""Regression tests for nested nullable MCP schema normalization (#133466)."""

from __future__ import annotations

import copy
import cProfile

import pytest

from tools.mcp_tool_schema import _normalize_mcp_input_schema


def _schema_for(depth: int, union_key: str) -> dict:
    schema = {
        "type": "object",
        "properties": {"label": {"type": "string"}},
        "required": ["label"],
    }
    for _ in range(depth):
        schema = {
            "type": "object",
            "required": [],
            "properties": {
                "child": {
                    union_key: [schema, {"type": "null"}],
                    "default": None,
                }
            },
        }
    return schema


def _profile_call_count(schema: dict) -> int:
    profiler = cProfile.Profile()
    profiler.runcall(_normalize_mcp_input_schema, schema)
    return sum(entry.callcount for entry in profiler.getstats())


@pytest.mark.parametrize("union_key", ("anyOf", "oneOf"))
def test_nested_nullable_schema_work_scales_with_schema_size(union_key: str):
    _normalize_mcp_input_schema({"type": "object"})
    shallow = _schema_for(16, union_key)
    deep = _schema_for(64, union_key)
    shallow_snapshot = copy.deepcopy(shallow)
    deep_snapshot = copy.deepcopy(deep)

    shallow_calls = _profile_call_count(shallow)
    deep_calls = _profile_call_count(deep)

    assert shallow == shallow_snapshot
    assert deep == deep_snapshot
    assert deep_calls <= shallow_calls * 5

    normalized = _normalize_mcp_input_schema(_schema_for(2, union_key))
    child = normalized["properties"]["child"]
    nested_child = child["properties"]["child"]
    assert child["nullable"] is True
    assert child["default"] is None
    assert nested_child["nullable"] is True
    assert nested_child["default"] is None
    assert nested_child["properties"]["label"] == {"type": "string"}
