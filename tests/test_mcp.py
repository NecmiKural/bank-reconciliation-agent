"""The MCP server exposes only read-only tools, computed and fenced like the agent's own."""

import os

import psycopg
import pytest
from mcp import Client

from recon import db, mcp_server, synth

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture(scope="module")
def dataset():
    try:
        conn = db.connect()
    except psycopg.OperationalError:
        if os.environ.get("CI"):
            raise
        pytest.skip("Postgres not reachable; run `docker compose up -d --wait`")
    ds = synth.generate(synth.SPECS[0])
    with conn:
        db.reset(conn, ds)
    return ds


async def test_every_tool_is_read_only(dataset):
    async with Client(mcp_server.build()) as client:
        listed = (await client.list_tools()).tools
    names = {tool.name for tool in listed}
    assert names == {"list_bank_lines", "get_bank_line", "search_candidates", "find_combinations"}
    assert all(t.annotations.read_only_hint and not t.annotations.destructive_hint for t in listed)


async def test_search_returns_the_labelled_invoice_with_its_relation(dataset):
    label = next(x for x in dataset.labels if x.case == "duplicate_amount")
    line = next(x for x in dataset.lines if x.id == label.line_id)
    async with Client(mcp_server.build()) as client:
        result = await client.call_tool(
            "search_candidates", {"line_id": line.id, "payee": line.payee, "invoice_number": ""}
        )
    assert not result.is_error
    candidates = result.structured_content["candidates"]
    truth = next(c for c in candidates if c["invoice"] == label.invoice_ids[0])
    assert truth["relation"] == "exact"


async def test_payer_text_comes_back_redacted_and_fenced(dataset):
    label = next(x for x in dataset.labels if x.pii and not x.attack_goal)
    async with Client(mcp_server.build()) as client:
        result = await client.call_tool("get_bank_line", {"line_id": label.line_id})
    text = str(result.structured_content)
    assert "<untrusted_payer_text>" in text
    assert not [value for _, value in label.pii if value in text]


async def test_combinations_are_found_in_code(dataset):
    label = next(x for x in dataset.labels if x.case == "split")
    async with Client(mcp_server.build()) as client:
        result = await client.call_tool(
            "find_combinations", {"line_id": label.line_id, "invoice_ids": list(label.invoice_ids)}
        )
    assert sorted(label.invoice_ids) in [
        sorted(c) for c in result.structured_content["exact_combinations"]
    ]


async def test_an_unknown_line_is_an_error_not_a_crash(dataset):
    async with Client(mcp_server.build()) as client:
        result = await client.call_tool("get_bank_line", {"line_id": "NOPE"})
    assert result.is_error


@pytest.fixture(scope="module")
def code_server(dataset):
    from test_sandbox import ready

    from recon import sandbox

    if not ready():
        if os.environ.get("CI"):
            pytest.fail(f"{sandbox.IMAGE} missing in CI")
        pytest.skip(f"Docker or {sandbox.IMAGE} not available; run `make sandbox-image`")
    return mcp_server.build_code_mode(deadline_s=20)


async def run_code(server, code):
    async with Client(server) as client:
        result = await client.call_tool("run_python", {"code": code})
    assert not result.is_error, result
    return result.structured_content


async def test_code_mode_serves_one_tool_and_reaches_the_read_tools(code_server):
    async with Client(code_server) as client:
        assert [t.name for t in (await client.list_tools()).tools] == ["run_python"]
    code = (
        "import json\n"
        "lines = json.loads(await list_bank_lines({'limit': 3}))['lines']\n"
        "found = json.loads(await search_candidates({'line_id': lines[0]['id'], 'payee': 'x'}))\n"
        "print(len(lines), 'candidates' in found)"
    )
    out = await run_code(code_server, code)
    assert out["error"] is None and out["stdout"].strip() == "3 True"


async def test_code_mode_output_carries_no_personal_data_or_host_secret(
    code_server, dataset, monkeypatch
):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-sentinel-not-a-real-key-000000")
    label = next(x for x in dataset.labels if x.pii and not x.attack_goal)
    code = (
        "import os\n"
        f"print(await get_bank_line({{'line_id': '{label.line_id}'}}))\n"
        "print(sorted(os.environ))"
    )
    out = await run_code(code_server, code)
    assert "<untrusted_payer_text>" in out["stdout"]
    assert not [value for _, value in label.pii if value in out["stdout"]]
    assert "sentinel" not in out["stdout"] and "DEEPSEEK_API_KEY" not in out["stdout"]


async def test_code_mode_has_no_way_to_write_and_checks_arguments(code_server):
    out = await run_code(code_server, "await queue_proposal({'line_id': 'A-L0001'})")
    assert "NameError" in out["error"]
    bad = "print(await search_candidates({'line_id': 'A-L0001', 'payee': 123}))"
    assert '"error"' in (await run_code(code_server, bad))["stdout"]
