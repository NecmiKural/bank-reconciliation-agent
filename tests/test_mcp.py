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
