"""Claude Code and Codex CLI clients (autoddg.llm.cli), run against fake CLIs

Set AUTODDG_TEST_CLAUDE_MODEL or AUTODDG_TEST_CODEX_MODEL to also make one real call through the
installed CLI (it uses that CLI's login).
"""

from __future__ import annotations

import json
import os
import sys
import textwrap

import pytest

from autoddg import AutoDDG, ClaudeCodeClient, CodexClient, LLMClient

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="fake CLIs are POSIX scripts")

MESSAGES = [
    {"role": "system", "content": "You describe datasets."},
    {"role": "user", "content": "Describe:\ncity,population\nLeeds,789194"},
]

FAKE_CLAUDE = """
import json, os, sys
args = sys.argv[1:]
prompt = sys.stdin.read()
if os.environ.get("FAKE_FAIL"):
    sys.stderr.write("login expired")
    sys.exit(1)
system = open(args[args.index("--system-prompt-file") + 1]).read()
answer = json.dumps({"args": args, "prompt": prompt, "system": system, "cwd": os.getcwd(),
                     "cwd_files": sorted(os.listdir("."))})
print("a line that is not JSON")
print(json.dumps({"type": "result", "subtype": "success",
                  "is_error": bool(os.environ.get("FAKE_ERROR")), "result": answer,
                  "usage": {"input_tokens": 3, "cache_creation_input_tokens": 1000,
                            "cache_read_input_tokens": 20, "output_tokens": 7}}))
"""

FAKE_CODEX = """
import json, os, sys
args = sys.argv[1:]
prompt = sys.stdin.read()
answer = json.dumps({"args": args, "prompt": prompt, "codex_home": os.environ.get("CODEX_HOME"),
                     "workdir_files": sorted(os.listdir(args[args.index("-C") + 1]))})
if not os.environ.get("FAKE_SILENT"):
    open(args[args.index("-o") + 1], "w").write(answer)
print(json.dumps({"type": "thread.started"}))
for n in (10, 5):
    print(json.dumps({"type": "turn.completed", "usage": {"input_tokens": n, "output_tokens": 2}}))
if os.environ.get("FAKE_SILENT"):
    print(json.dumps({"type": "turn.failed", "error": {"message": "usage limit"}}))
"""


def fake_cli(tmp_path, name, source):
    path = tmp_path / name
    path.write_text(f"#!{sys.executable}\n" + textwrap.dedent(source))
    path.chmod(0o755)
    return str(path)


def answer_of(response):
    return json.loads(response["choices"][0]["message"]["content"])


def test_claude_call(tmp_path):
    client = ClaudeCodeClient(fake_cli(tmp_path, "claude", FAKE_CLAUDE), effort="low")
    response = client.chat_completions_create(model="claude-sonnet-5-5", messages=MESSAGES)
    answer = answer_of(response)
    assert answer["system"] == "You describe datasets."
    assert answer["prompt"] == MESSAGES[1]["content"]
    args = answer["args"]
    assert args[:3] == ["-p", "--output-format", "json"]
    assert args[args.index("--tools") + 1] == ""  # no tools
    assert "--safe-mode" in args and "--no-session-persistence" in args
    assert args[args.index("--model") + 1] == "claude-sonnet-5-5"
    assert args[args.index("--effort") + 1] == "low"
    assert answer["cwd"] != os.getcwd() and answer["cwd_files"] == ["system_prompt.txt"]
    assert response["usage"] == {
        "prompt_tokens": 1023,
        "completion_tokens": 7,
        "total_tokens": 1030,
    }


def test_claude_without_system_message_or_model(tmp_path):
    client = ClaudeCodeClient(fake_cli(tmp_path, "claude", FAKE_CLAUDE))
    answer = answer_of(client.chat_completions_create(model="", messages=MESSAGES[1:]))
    assert answer["system"] == "You are a helpful assistant."
    assert "--model" not in answer["args"] and "--effort" not in answer["args"]


def test_claude_errors(tmp_path):
    client = ClaudeCodeClient(fake_cli(tmp_path, "claude", FAKE_CLAUDE), env={"FAKE_ERROR": "1"})
    with pytest.raises(RuntimeError, match="reported an error"):
        client.chat_completions_create(model="m", messages=MESSAGES)
    client = ClaudeCodeClient(fake_cli(tmp_path, "claude", FAKE_CLAUDE), env={"FAKE_FAIL": "1"})
    with pytest.raises(RuntimeError, match="login expired"):
        client.chat_completions_create(model="m", messages=MESSAGES)


def test_codex_call(tmp_path):
    home = str(tmp_path / "codex-home")
    client = CodexClient(fake_cli(tmp_path, "codex", FAKE_CODEX), effort="high", codex_home=home)
    response = client.chat_completions_create(model="gpt-6.1-sol", messages=MESSAGES)
    answer = answer_of(response)
    assert answer["prompt"] == "You describe datasets.\n\n" + MESSAGES[1]["content"]
    args = answer["args"]
    assert args[0] == "exec" and args[-1] == "-"
    assert args[args.index("--sandbox") + 1] == "read-only"
    assert "--ephemeral" in args and "--ignore-user-config" in args
    assert args[args.index("-m") + 1] == "gpt-6.1-sol"
    assert 'model_reasoning_effort="high"' in args
    assert answer["codex_home"] == home and answer["workdir_files"] == []
    assert response["usage"] == {"prompt_tokens": 15, "completion_tokens": 4, "total_tokens": 19}


def test_codex_without_answer(tmp_path):
    client = CodexClient(fake_cli(tmp_path, "codex", FAKE_CODEX), env={"FAKE_SILENT": "1"})
    with pytest.raises(RuntimeError, match="usage limit"):
        client.chat_completions_create(model="m", messages=MESSAGES)


def test_timeout(tmp_path):
    slow = fake_cli(tmp_path, "claude", "import time; time.sleep(5)")
    with pytest.raises(TimeoutError):
        ClaudeCodeClient(slow, timeout=0.5).chat_completions_create(model="m", messages=MESSAGES)


def test_missing_executable():
    with pytest.raises(FileNotFoundError):
        ClaudeCodeClient("no-such-claude-binary")


def test_multi_turn_messages_are_rendered(tmp_path):
    client = ClaudeCodeClient(fake_cli(tmp_path, "claude", FAKE_CLAUDE))
    messages = [
        *MESSAGES,
        {"role": "assistant", "content": "A city."},
        {"role": "user", "content": "More."},
    ]
    prompt = answer_of(client.chat_completions_create(model="m", messages=messages))["prompt"]
    assert prompt.startswith("User: Describe:") and prompt.endswith(
        "Assistant: A city.\n\nUser: More."
    )


def test_autoddg_uses_an_llm_client_as_is(tmp_path):
    client = ClaudeCodeClient(fake_cli(tmp_path, "claude", FAKE_CLAUDE))
    autoddg = AutoDDG(client=client, model_name="claude-sonnet-5-5")
    assert autoddg.description_generator.llm_client is client
    _, description = autoddg.describe_dataset(dataset_sample="city,population\nLeeds,789194")
    assert "Leeds" in json.loads(description)["prompt"]
    assert isinstance(client, LLMClient)


@pytest.mark.skipif(not os.environ.get("AUTODDG_TEST_CLAUDE_MODEL"), reason="real Claude Code call")
def test_real_claude():
    client = ClaudeCodeClient()
    model = os.environ["AUTODDG_TEST_CLAUDE_MODEL"]
    response = client.chat_completions_create(
        model=model, messages=[{"role": "user", "content": "Reply with exactly: OK"}]
    )
    assert response["choices"][0]["message"]["content"].strip() == "OK"


@pytest.mark.skipif(not os.environ.get("AUTODDG_TEST_CODEX_MODEL"), reason="real Codex call")
def test_real_codex():
    client = CodexClient(os.environ.get("AUTODDG_TEST_CODEX_BINARY", "codex"))
    model = os.environ["AUTODDG_TEST_CODEX_MODEL"]
    response = client.chat_completions_create(
        model=model, messages=[{"role": "user", "content": "Reply with exactly: OK"}]
    )
    assert response["choices"][0]["message"]["content"].strip() == "OK"
