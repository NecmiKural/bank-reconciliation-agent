"""The pre-registered verdicts follow the thresholds written before the frozen runs."""

from test_report import code_mode_results

from recon import hypotheses


def test_a_cheaper_cell_with_few_passes_holds_h1():
    direct, code = code_mode_results()
    table, held = hypotheses.h1_to_h3([direct, code])
    assert held["H1"] == {"deepseek-flash": [True]}
    assert "| 20.0 (54.0) | -60%; $0.0200 against $0.0500 | holds |" in table[2]


def test_one_code_run_dearer_than_the_cheapest_per_line_run_fails_h1():
    direct, code = code_mode_results()
    code["runs"].append({**code["runs"][0], "summary": {**code["runs"][0]["summary"]}})
    code["runs"][1]["summary"]["cost_per_100_bank_lines_usd"] = 0.06
    _, held = hypotheses.h1_to_h3([direct, code])
    assert held["H1"] == {"deepseek-flash": [False]}


def test_too_many_model_passes_fail_h1():
    direct, code = code_mode_results()
    code["runs"][0]["summary"]["model_requests_per_100_agent_lines"] = 60.0
    _, held = hypotheses.h1_to_h3([direct, code])
    assert held["H1"] == {"deepseek-flash": [False]}
