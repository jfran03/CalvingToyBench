"""Zero-shot Harbor agent: one API call per window, no tools.

Harbor hands us the task's instruction.md (the exact bench.py prompt). We send it to the model
once, parse {"probability", "reason"}, and write that to /app/answer.json inside the sandbox.
The verifier then scores the file against the hidden label.

This is an "external" agent (BaseAgent): the loop runs on your machine, not in the container,
so the API key never enters the sandbox and the sandbox can stay offline.

Run (from the project root, see harbor_run.py):
  harbor run -p harbor_tasks/both -a harbor_agent:ZeroShotAgent -m anthropic/claude-sonnet-5-5 --env-file .env
Supported providers: anthropic, openai, mock (offline, no key, tests the plumbing).
"""
import asyncio, json, tempfile
from pathlib import Path

import requests
from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

from bench import CALL, parse  # same API calls, system prompt and JSON parser as bench.py

PROVIDERS = ("anthropic", "openai", "mock")
ANSWER_PATH = "/app/answer.json"


class ZeroShotAgent(BaseAgent):
    @staticmethod
    def name() -> str:
        return "zero-shot"

    def version(self) -> str:
        return "1.0.0"

    async def setup(self, environment: BaseEnvironment) -> None:
        pass

    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        provider, _, model = (self.model_name or "").partition("/")
        if provider not in PROVIDERS or not model:
            raise ValueError(f"-m must be <provider>/<model> with provider in {PROVIDERS}, got {self.model_name!r}")

        # strip(): instruction.md ends with a newline that bench.py's prompt doesn't have
        answer, log = await asyncio.to_thread(self._ask, provider, model, instruction.strip())

        usage = log.get("usage") or {}
        context.n_input_tokens = usage.get("input_tokens")
        context.n_output_tokens = usage.get("output_tokens")
        context.metadata = {"attempts": log["attempts"], "error": log.get("error")}
        (self.logs_dir / "response.json").write_text(json.dumps(log, indent=2))

        await environment.exec("mkdir -p /app")
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "answer.json"
            f.write_text(json.dumps(answer))
            await environment.upload_file(f, ANSWER_PATH)

    def _ask(self, provider, model, prompt):
        """Same retry policy as bench.run_one: fail fast on auth/bad-request errors, back off on the rest."""
        log = {"provider": provider, "model": model, "attempts": 0}
        for attempt in range(6):
            log["attempts"] = attempt + 1
            try:
                text, usage = CALL[provider](model, prompt)
                log.update(raw=text, usage=usage)
                p, reason = parse(text)
                return {"probability": p, "reason": reason}, log
            except requests.HTTPError as e:
                code = e.response.status_code
                if code in (400, 401, 403, 404):
                    raise RuntimeError(f"{provider} error {code}: {e.response.text[:500]}") from e
                log["error"] = f"HTTP {code}"
                _sleep(2 ** attempt * 3)
            except Exception as e:  # unparseable reply, timeout, connection reset
                log["error"] = str(e)[:300]
                if attempt < 5:
                    _sleep(2 ** attempt)
        # Gave up: write an explicit "no answer" so the verifier marks it invalid (prob=None in score.py).
        return {"probability": None, "reason": None, "error": log["error"]}, log


def _sleep(s):
    import time
    time.sleep(s)
