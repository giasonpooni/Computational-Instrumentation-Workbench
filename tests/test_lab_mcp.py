import asyncio
import json

import pytest

pytest.importorskip("mcp")

from mcp import Client  # noqa: E402

from ciw.lab import runner  # noqa: E402
from ciw.lab.mcp_server import build_server  # noqa: E402

EXPECTED = {"ciw_lab_list_tasks", "ciw_lab_get_report", "ciw_lab_plan_next", "ciw_lab_run_tasks",
            "ciw_lab_verify_run", "ciw_lab_classify_workspace", "ciw_lab_explain_labels"}


def _call(server, name, arguments=None):
    async def go():
        async with Client(server) as client:
            result = await client.call_tool(name, arguments or {})
            return result.is_error, "".join(part.text for part in result.content if hasattr(part, "text"))
    return asyncio.run(go())


def test_tools_are_declared_and_no_tool_accepts_labels(tmp_path):
    server = build_server(None, tmp_path / "work")

    async def go():
        async with Client(server) as client:
            return (await client.list_tools()).tools
    tools = {tool.name: tool for tool in asyncio.run(go())}
    assert set(tools) == EXPECTED
    for tool in tools.values():
        properties = set(tool.input_schema.get("properties", {}))
        assert not properties & {"label", "labels", "evidence_status", "finding", "findings", "report", "basis"}
    assert tools["ciw_lab_run_tasks"].annotations.read_only_hint is False
    assert all(tools[name].annotations.read_only_hint for name in EXPECTED - {"ciw_lab_run_tasks"})


def test_run_read_plan_and_verify_through_mcp(tmp_path):
    retained = tmp_path / "retained"
    runner.run_queue(retained, ["T156"])
    server = build_server(retained, tmp_path / "work")
    error, text = _call(server, "ciw_lab_list_tasks", {"section": "research-portfolio", "response_format": "json"})
    listing = json.loads(text)
    assert not error and listing["total"] == 14 and listing["items"][1]["task_id"] == "T156"
    error, text = _call(server, "ciw_lab_run_tasks", {"task_ids": ["T156"]})
    assert not error and "T156" in text and "physical validation not_established" in text
    error, text = _call(server, "ciw_lab_get_report", {"task_id": "T156"})
    assert not error and "**Evidence status:**" in text and str(tmp_path / "work") in text
    error, text = _call(server, "ciw_lab_verify_run")
    assert not error and json.loads(text)["passed"] is True
    error, text = _call(server, "ciw_lab_plan_next", {"limit": 3, "response_format": "json"})
    assert not error and json.loads(text)["schema"] == "ciw.lab-next.v1"
    error, text = _call(server, "ciw_lab_explain_labels")
    assert "independently_verified" in text and "Machine safety" in text


def test_errors_are_actionable(tmp_path):
    server = build_server(None, tmp_path / "work")
    error, text = _call(server, "ciw_lab_run_tasks", {"task_ids": ["T1"]})
    assert error and "T003" in text
    error, text = _call(server, "ciw_lab_get_report", {"task_id": "T010"})
    assert error and "ciw_lab_run_tasks" in text
    error, text = _call(server, "ciw_lab_verify_run")
    assert error and "retained" in text
    error, text = _call(server, "ciw_lab_list_tasks", {"limit": 0})
    assert error and "limit" in text


def test_stdio_entry_point_serves_the_queue(tmp_path):
    import os
    import sys
    from pathlib import Path

    from mcp import StdioServerParameters

    source = str(Path(__file__).resolve().parents[1] / "src")
    environment = {**os.environ, "PYTHONPATH": source + os.pathsep + os.environ.get("PYTHONPATH", "")}
    parameters = StdioServerParameters(command=sys.executable, env=environment,
                                       args=["-m", "ciw", "lab", "mcp", "--workdir", str(tmp_path / "work")])

    async def go():
        async with Client(parameters) as client:
            names = {tool.name for tool in (await client.list_tools()).tools}
            result = await client.call_tool("ciw_lab_list_tasks", {"limit": 2, "response_format": "json"})
            return names, json.loads(result.content[0].text)
    names, listing = asyncio.run(go())
    assert names == EXPECTED and listing["total"] == 168 and listing["has_more"] is True
