"""Ask a fresh Codex or Claude Code session which repository skill it would load for each trigger case.

Cases live in ``.agents/skills/<name>/evals/evals.json``. A ``<name>-hit-*`` case passes when the agent selects that
skill; a ``<name>-miss-*`` case passes when it selects anything else. Cases marked ``explicit`` invoke an
explicit-only skill by name; Claude Code runs those as user slash commands, so the Claude run skips them. The run
calls a paid model, so it is a manual check after editing skill descriptions, not a CI step.

    python tools/eval_skill_triggers.py --agent codex
    python tools/eval_skill_triggers.py --agent claude --skill benchmark
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODEX_APP_CLI = Path("/Applications/Codex.app/Contents/Resources/codex-cli/bin/codex")
ANSWER_LINE = re.compile(r"^\s*([a-z0-9-]+-(?:hit|miss)-\d+)\s*:\s*([a-z0-9-]+)\s*$", re.MULTILINE)


def load_cases(skills: list[str]) -> list[dict[str, str]]:
    cases = []
    for path in sorted((ROOT / ".agents/skills").glob("*/evals/evals.json")):
        if not skills or path.parents[1].name in skills:
            cases.extend(json.loads(path.read_text(encoding="utf-8")))
    return cases


def build_prompt(cases: list[dict[str, str]]) -> str:
    requests = "\n".join(f"{case['id']}: {case['prompt']}" for case in cases)
    return (
        "Do not run tools, read files, or do any of the tasks. For each request below, answer with the single "
        "repository skill you would load first, or 'none'. Reply only with lines formatted '<id>: <skill-name>'.\n\n"
        + requests
    )


def agent_command(agent: str, prompt: str) -> list[str]:
    if agent == "claude":
        return ["claude", "-p", prompt, "--max-turns", "1"]
    codex = shutil.which("codex") or str(CODEX_APP_CLI)
    return [codex, "exec", "--skip-git-repo-check", "-s", "read-only", prompt]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--agent", choices=("codex", "claude"), required=True)
    parser.add_argument("--skill", action="append", default=[], help="limit to this skill (repeatable)")
    args = parser.parse_args(argv)

    cases = [case for case in load_cases(args.skill) if not (args.agent == "claude" and case.get("explicit"))]
    completed = subprocess.run(  # noqa: S603 - argv is a fixed agent CLI plus a prompt built from tracked eval files.
        agent_command(args.agent, build_prompt(cases)),
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    answers = dict(ANSWER_LINE.findall(completed.stdout))

    failures = 0
    for case in cases:
        expected_skill = case["id"].rsplit("-", 2)[0]
        selected = answers.get(case["id"], "<no answer>")
        passed = selected == expected_skill if "-hit-" in case["id"] else selected != expected_skill
        failures += not passed
        print(f"{'PASS' if passed else 'FAIL'}  {case['id']}: expected {case['expected_output']}, got {selected}")
    print(f"{len(cases) - failures}/{len(cases)} passed")
    return int(bool(failures))


if __name__ == "__main__":
    sys.exit(main())
