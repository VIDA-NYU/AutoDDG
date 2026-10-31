from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import warnings
from pathlib import Path
from typing import Any

from beartype import beartype

from .base import LLMClient

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."


def _split_messages(messages: list[dict[str, str]]) -> tuple[str, str]:
    """
    Split chat messages into the system text and one prompt for a single-turn CLI call

    Args:
        messages: List of message dicts with 'role' and 'content' keys

    Returns:
        (system_text, prompt); a lone user message is passed as it is, longer exchanges are
        rendered as "Role: content" turns
    """
    system = "\n\n".join(m.get("content", "") for m in messages if m.get("role") == "system")
    turns = [m for m in messages if m.get("role") != "system"]
    if len(turns) == 1 and turns[0].get("role") == "user":
        return system, turns[0].get("content", "")
    prompt = "\n\n".join(
        f"{m.get('role', 'user').capitalize()}: {m.get('content', '')}" for m in turns
    )
    return system, prompt


def _completion(content: str, prompt_tokens: int, completion_tokens: int) -> dict[str, Any]:
    return {
        "choices": [{"message": {"role": "assistant", "content": content}}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


@beartype
class CLIAgentClient(LLMClient):
    """Base class for coding-agent CLIs used as a plain LLM: one headless call per completion"""

    def __init__(
        self,
        executable: str,
        *,
        timeout: float = 600.0,
        extra_args: list[str] | None = None,
        env: dict[str, str] | None = None,
    ) -> None:
        """
        Args:
            executable: Name or path of the CLI
            timeout: Seconds allowed for one call
            extra_args: Additional command-line arguments, added before the prompt
            env: Environment variables to set for the CLI, on top of the current environment
        """
        path = shutil.which(executable)
        if path is None:
            raise FileNotFoundError(f"{executable!r} not found; install it or pass its full path")
        self.executable = path
        self.timeout = timeout
        self.extra_args = list(extra_args or [])
        self.env = dict(env or {})

    def _run(self, command: list[str], prompt: str, cwd: str) -> subprocess.CompletedProcess:
        """Run the CLI once with the prompt on stdin"""
        try:
            completed = subprocess.run(
                command,
                input=prompt,
                capture_output=True,
                text=True,
                encoding="utf-8",
                cwd=cwd,
                env={**os.environ, **self.env},
                timeout=self.timeout,
            )
        except subprocess.TimeoutExpired as error:
            raise TimeoutError(
                f"{Path(self.executable).name} did not answer within {self.timeout:g} s"
            ) from error
        if completed.returncode != 0:
            raise RuntimeError(
                f"{Path(self.executable).name} exited with code {completed.returncode}: "
                f"{(completed.stderr or completed.stdout).strip()[-2000:]}"
            )
        return completed

    @staticmethod
    def _warn_temperature(temperature: float) -> None:
        if temperature != 0.0:
            warnings.warn(
                "CLI clients do not support a sampling temperature; it is ignored",
                stacklevel=3,
            )


@beartype
class ClaudeCodeClient(CLIAgentClient):
    """
    Claude Code as the LLM: one `claude -p` call per completion

    Uses the CLI's own login (a Claude subscription or ANTHROPIC_API_KEY), so no API client is
    needed. Each call runs in an empty temporary directory with no tools, no saved session and
    --safe-mode, so the user's CLAUDE.md, hooks, plugins, skills and MCP servers stay out of the
    prompt. The system message replaces Claude Code's own system prompt.

    Examples:
        >>> from autoddg import AutoDDG, ClaudeCodeClient
        >>> autoddg = AutoDDG(client=ClaudeCodeClient(), model_name="claude-sonnet-5-5")
    """

    def __init__(
        self,
        executable: str = "claude",
        *,
        effort: str | None = None,
        timeout: float = 600.0,
        extra_args: list[str] | None = None,
        env: dict[str, str] | None = None,
    ) -> None:
        """
        Args:
            executable: Name or path of the `claude` CLI
            effort: Value for --effort (e.g. "low", "medium", "high"); None keeps the CLI default
            timeout: Seconds allowed for one call
            extra_args: Additional command-line arguments
            env: Environment variables to set for the CLI
        """
        super().__init__(executable, timeout=timeout, extra_args=extra_args, env=env)
        self.effort = effort

    def chat_completions_create(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Generate a chat completion with `claude -p`

        Args:
            model: Model name or alias passed to --model (e.g. "claude-sonnet-5-5", "sonnet");
                an empty string keeps the CLI default
            messages: List of message dicts with 'role' and 'content' keys
            temperature: Ignored (the CLI has no temperature setting); a warning is issued if
                it is not 0
            **kwargs: Ignored (for compatibility with other clients)

        Returns:
            Response dict compatible with OpenAI API format
        """
        self._warn_temperature(temperature)
        system, prompt = _split_messages(messages)
        with tempfile.TemporaryDirectory() as cwd:
            system_file = Path(cwd) / "system_prompt.txt"
            system_file.write_text(system or DEFAULT_SYSTEM_PROMPT, encoding="utf-8")
            command = [
                self.executable,
                "-p",
                "--output-format",
                "json",
                "--tools",
                "",
                "--safe-mode",
                "--no-session-persistence",
                "--system-prompt-file",
                str(system_file),
            ]
            if model:
                command += ["--model", model]
            if self.effort:
                command += ["--effort", self.effort]
            completed = self._run(command + self.extra_args, prompt, cwd)

        result = None
        for line in completed.stdout.splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict) and event.get("type") == "result":
                result = event
        if result is None:
            raise RuntimeError(f"claude printed no result: {completed.stdout.strip()[-2000:]}")
        if result.get("is_error") or result.get("subtype") != "success":
            raise RuntimeError(f"claude reported an error: {json.dumps(result)[:2000]}")

        usage = result.get("usage") or {}
        prompt_tokens = sum(
            usage.get(key) or 0
            for key in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")
        )
        return _completion(
            result.get("result") or "", prompt_tokens, usage.get("output_tokens") or 0
        )


@beartype
class CodexClient(CLIAgentClient):
    """
    Codex as the LLM: one `codex exec` run per completion

    Uses the CLI's own login (a ChatGPT plan or an API key), so no API client is needed. Each call
    is an ephemeral run in an empty temporary directory with a read-only sandbox, without the
    user's config.toml. Codex has no system-prompt option, so the system message is placed before
    the prompt. Codex still reads AGENTS.md from its home directory ($CODEX_HOME, by default
    ~/.codex); pass `codex_home` to use a separate home, logged in once with
    `CODEX_HOME=<dir> codex login`.

    Examples:
        >>> from autoddg import AutoDDG, CodexClient
        >>> autoddg = AutoDDG(client=CodexClient(), model_name="gpt-6.1-sol")
    """

    def __init__(
        self,
        executable: str = "codex",
        *,
        effort: str | None = None,
        codex_home: str | None = None,
        timeout: float = 600.0,
        extra_args: list[str] | None = None,
        env: dict[str, str] | None = None,
    ) -> None:
        """
        Args:
            executable: Name or path of the `codex` CLI
            effort: Value for model_reasoning_effort (e.g. "low", "medium", "high"); None keeps
                the model default
            codex_home: Codex home directory to use instead of $CODEX_HOME or ~/.codex
            timeout: Seconds allowed for one call
            extra_args: Additional `codex exec` arguments
            env: Environment variables to set for the CLI
        """
        environment = dict(env or {})
        if codex_home is not None:
            environment["CODEX_HOME"] = codex_home
        super().__init__(executable, timeout=timeout, extra_args=extra_args, env=environment)
        self.effort = effort

    def chat_completions_create(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Generate a chat completion with `codex exec`

        Args:
            model: Model name passed to -m; an empty string keeps the CLI default
            messages: List of message dicts with 'role' and 'content' keys
            temperature: Ignored (the CLI has no temperature setting); a warning is issued if
                it is not 0
            **kwargs: Ignored (for compatibility with other clients)

        Returns:
            Response dict compatible with OpenAI API format
        """
        self._warn_temperature(temperature)
        system, prompt = _split_messages(messages)
        text = f"{system}\n\n{prompt}" if system else prompt
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp) / "work"
            workdir.mkdir()
            last_message = Path(tmp) / "last_message.txt"
            command = [
                self.executable,
                "exec",
                "--json",
                "--skip-git-repo-check",
                "--ephemeral",
                "--ignore-user-config",
                "--sandbox",
                "read-only",
                "-C",
                str(workdir),
                "-o",
                str(last_message),
            ]
            if model:
                command += ["-m", model]
            if self.effort:
                command += ["-c", f'model_reasoning_effort="{self.effort}"']
            completed = self._run(command + self.extra_args + ["-"], text, tmp)
            content = (
                last_message.read_text(encoding="utf-8").strip() if last_message.exists() else ""
            )

        prompt_tokens = completion_tokens = 0
        errors = []
        for line in completed.stdout.splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict):
                continue
            if event.get("type") == "turn.completed":
                usage = event.get("usage") or {}
                prompt_tokens += usage.get("input_tokens") or 0
                completion_tokens += usage.get("output_tokens") or 0
            elif event.get("type") in ("error", "turn.failed"):
                errors.append(json.dumps(event)[:500])
        if not content:
            raise RuntimeError(f"codex gave no answer: {'; '.join(errors) or 'empty last message'}")
        return _completion(content, prompt_tokens, completion_tokens)
