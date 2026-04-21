from __future__ import annotations
import json, time, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterable, Any
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel

from .client import LMClient
from .settings import Settings, Phase
from .tools import Registry, build_registry
from .context import ProjectContext

SYSTEM_PROMPT = """You are **Coder**, a senior software engineer working directly on the user's machine.

You have tools to read, write, edit, delete files; scaffold projects; run tests, linters, shell commands; manage git and GitHub; browse the web; persist memory; and spawn focused sub-agents. Use them. Do not just describe what you would do — do it, verify it, and report the diff.

Principles:
- Plan briefly (use `todo_write`), then execute. Small, verified steps beat big speculative ones.
- For non-trivial tasks, call `todo_write` first, then keep it updated as you progress.
- Always check before you change (read_file, grep, ast_symbols, run_tests).
- After each edit, verify (run_tests, run_typecheck, run_lint). Fix what you broke.
- When the user asks for a whole project, scaffold it end-to-end: init repo, deps, sample code, tests, CI, README.
- Never invent APIs. If unsure, read the source.
- Prefer `apply_patch` / `multi_edit` for multi-site edits — atomic and robust.
- When a task is read-heavy (many greps/reads), spawn a sub-agent via `spawn_subagent` so the main context stays clean.
- Use `think` for non-trivial debugging or planning — it is free of side effects but keeps your reasoning auditable.
- Only ask the user for clarification when a true ambiguity blocks you. Otherwise decide and proceed.
- Respond in the user's language.

Tool usage is encouraged. **Emit multiple parallel tool calls in one turn** when the information is independent — the runtime fans them out concurrently.
"""


COMPACTION_SYSTEM = """You are summarizing a coding-agent transcript so it fits a smaller context window.
Preserve:
- The user's original goal
- Decisions already made and why
- Files/paths/symbols already inspected and the key facts learned
- Outstanding TODOs
Discard raw tool output, verbose logs, and redundant chatter. Be factual and concise (<= 800 tokens)."""


class BudgetExceeded(Exception):
    pass


class Agent:
    def __init__(self, settings: Settings | None = None):
        self.s = settings or Settings()
        self.client = LMClient(self.s)
        self.registry: Registry = build_registry(self.s)
        self.registry.attach_agent(self)
        self.ctx = ProjectContext(self.s)
        self.console = Console()
        self._plan: list[dict] = []
        self._scratchpad: list[str] = []
        self._started_at: float = 0.0
        self._tool_calls_made: int = 0

    # ---------- plan / scratchpad (exposed to tools) ----------

    def set_plan(self, tasks: list[dict]) -> None:
        self._plan = tasks

    def get_plan(self) -> list[dict]:
        return self._plan

    def add_thought(self, text: str) -> None:
        self._scratchpad.append(text)

    def budget_status(self) -> dict:
        elapsed = time.time() - self._started_at if self._started_at else 0.0
        return {
            "tool_calls": self._tool_calls_made,
            "steps_limit": self.s.max_tool_steps,
            "elapsed_sec": round(elapsed, 1),
            "wallclock_budget_sec": self.s.wallclock_budget_sec,
        }

    # ---------- message bootstrap ----------

    def _plan_block(self) -> str:
        if not self._plan:
            return ""
        rows = []
        for t in self._plan:
            box = {"pending": "[ ]", "in_progress": "[~]", "completed": "[x]"}.get(t.get("status", "pending"), "[ ]")
            rows.append(f"{box} {t.get('id','?')}: {t.get('content','')}")
        return "\n## Current plan\n" + "\n".join(rows)

    def _bootstrap_messages(self, user_msg: str) -> list[dict]:
        project_summary = self.ctx.summarize()
        sys_second = (
            f"Current working directory: {self.s.workdir}\n"
            f"Autonomy level: {self.s.autonomy}\n"
            f"Parallel tool calls: {'on' if self.s.parallel_tools else 'off'}\n"
            f"{self._plan_block()}\n\n"
            f"{project_summary}"
        )
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "system", "content": sys_second},
            {"role": "user", "content": user_msg},
        ]

    # ---------- tool fan-out ----------

    def _dispatch_one(self, tc: dict) -> tuple[dict, str]:
        name = tc["function"]["name"]
        args = tc["function"]["arguments"]
        return tc, self.registry.dispatch(name, args)

    def _run_tool_calls(self, tool_calls: list[dict], step: int) -> list[dict]:
        results: list[dict] = []
        self._tool_calls_made += len(tool_calls)

        def _render_call(tc: dict) -> None:
            self.console.print(Panel.fit(
                f"[bold cyan]{tc['function']['name']}[/]\n{tc['function']['arguments'][:500]}",
                title=f"tool call [{step}/{self.s.max_tool_steps}]",
            ))

        def _render_result(tc: dict, result: str) -> None:
            self.console.print(Panel.fit(
                result[:1200] + ("…" if len(result) > 1200 else ""),
                title=f"result · {tc['function']['name']}",
                border_style="green",
            ))

        if self.s.parallel_tools and len(tool_calls) > 1:
            with ThreadPoolExecutor(max_workers=self.s.parallel_max_workers) as ex:
                futures = {ex.submit(self._dispatch_one, tc): tc for tc in tool_calls}
                for tc in tool_calls:
                    _render_call(tc)
                for fut in as_completed(futures):
                    tc, result = fut.result()
                    _render_result(tc, result)
                    results.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": result,
                    })
            order = {tc["id"]: i for i, tc in enumerate(tool_calls)}
            results.sort(key=lambda m: order.get(m["tool_call_id"], 0))
        else:
            for tc in tool_calls:
                _render_call(tc)
                _, result = self._dispatch_one(tc)
                _render_result(tc, result)
                results.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": result,
                })
        return results

    # ---------- compaction ----------

    def _maybe_compact(self, messages: list[dict]) -> list[dict]:
        if not self.s.compaction_enabled:
            return messages
        toks = self.client.count_tokens_rough(messages)
        limit = int(self.s.compaction_max_context_tokens * self.s.compaction_trigger_ratio)
        if toks < limit:
            return messages

        keep_head = [m for m in messages if m["role"] == "system"][:2]
        tail = messages[-self.s.compaction_keep_recent_turns * 2:]
        middle = messages[len(keep_head):-(self.s.compaction_keep_recent_turns * 2)] if len(messages) > (len(keep_head) + self.s.compaction_keep_recent_turns * 2) else []
        if not middle:
            return messages

        summary_prompt = [
            {"role": "system", "content": COMPACTION_SYSTEM},
            {"role": "user", "content": json.dumps(middle, default=str)[:60000]},
        ]
        try:
            resp = self.client.chat(summary_prompt, phase="reflection", prefer_small=True)
            summary = resp.choices[0].message.content or ""
        except Exception as e:
            self.console.print(f"[yellow]compaction failed: {e}[/]")
            return messages

        self.console.print(Panel.fit(
            f"context compacted ({toks} tok rough → summarized {len(middle)} msgs)",
            border_style="yellow",
        ))
        return keep_head + [
            {"role": "system", "content": f"## Conversation summary (earlier turns)\n{summary}"}
        ] + tail

    # ---------- reflection on tool error ----------

    @staticmethod
    def _tool_result_failed(payload: str) -> bool:
        try:
            obj = json.loads(payload)
        except Exception:
            return False
        if not isinstance(obj, dict):
            return False
        return obj.get("ok") is False or "error" in obj

    def _reflect_on_failures(self, tool_results: list[dict]) -> dict | None:
        failures = [r for r in tool_results if self._tool_result_failed(r.get("content", ""))]
        if not failures or not self.s.reflect_on_error:
            return None
        names = []
        for r in failures:
            try:
                names.append(json.loads(r["content"]).get("error", "error"))
            except Exception:
                names.append("error")
        return {
            "role": "system",
            "content": (
                "Previous tool call(s) failed: " + "; ".join(names[:4]) +
                ". Reconsider your approach: verify preconditions, "
                "read the current state, or try an alternative tool. "
                "Do not retry the exact same call with the exact same arguments."
            ),
        }

    # ---------- model phase heuristic ----------

    def _phase_for_step(self, step: int) -> Phase:
        if step == 0:
            return "planning"
        return "execution"

    # ---------- run loop ----------

    def run(self, user_msg: str) -> str:
        self._started_at = time.time()
        self._tool_calls_made = 0
        messages = self._bootstrap_messages(user_msg)
        tools = self.registry.to_openai()

        final_text = ""
        for step in range(self.s.max_tool_steps):
            if self.s.wallclock_budget_sec and (time.time() - self._started_at) > self.s.wallclock_budget_sec:
                final_text = "[wall-clock budget exceeded]"
                break

            messages = self._maybe_compact(messages)
            phase = self._phase_for_step(step)

            if self.s.stream:
                buf: list[str] = []
                with Live(Panel("", title=f"coder · streaming [{step+1}]", border_style="magenta"),
                          console=self.console, refresh_per_second=12, transient=True) as live:
                    def on_tok_live(t: str) -> None:
                        buf.append(t)
                        live.update(Panel(Markdown("".join(buf)[-4000:]),
                                          title=f"coder · streaming [{step+1}]",
                                          border_style="magenta"))
                    assembled = self.client.stream_assembled(
                        messages, tools=tools, phase=phase, on_token=on_tok_live,
                    )
                msg_content = assembled["content"]
                tool_calls = assembled["tool_calls"]
            else:
                resp = self.client.chat(messages, tools=tools, phase=phase)
                m = resp.choices[0].message
                msg_content = m.content or ""
                tool_calls = [
                    {"id": tc.id, "type": "function",
                     "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                    for tc in (getattr(m, "tool_calls", None) or [])
                ]

            if tool_calls:
                messages.append({
                    "role": "assistant",
                    "content": msg_content or "",
                    "tool_calls": tool_calls,
                })
                tool_results = self._run_tool_calls(tool_calls, step + 1)
                messages.extend(tool_results)

                reflection = self._reflect_on_failures(tool_results)
                if reflection is not None:
                    messages.append(reflection)
                continue

            final_text = msg_content
            break
        else:
            final_text = "[tool step limit reached]"

        self.console.print(Panel(Markdown(final_text or "(no output)"), title="coder", border_style="magenta"))
        stats = self.budget_status()
        self.console.print(
            f"[dim]tool calls: {stats['tool_calls']} · elapsed: {stats['elapsed_sec']}s[/]"
        )
        return final_text
