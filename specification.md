PsySafe AI SDK – Technical Specification Brief
(v 0.1 – Draft for internal discussion)

1 Vision & Scope
PsySafe AI is an open-source Python SDK that helps teams embed psychological-safety guardrails into any LLM workflow—local or hosted, sync or async, chat or single-shot.
The SDK’s job is to make “doing the right thing” the path of least resistance for Python developers.

2 Design Principles
#	Principle	Practical Meaning for the SDK
1	Prompt-first, code-light	A guardrail should be enabled with one line of code (or decorator) and zero extra infra.
2	Type-safety everywhere	Public APIs are fully type-hinted; Pydantic models validate inputs & outputs.
3	Framework-agnostic	No hard dependency on LangChain, LlamaIndex, FastAPI, etc. – optional adapters only.
4	Composable & override-able	Guardrails can be layered, combined, partially overridden, or disabled by config.
5	No network surprises	Core library is offline-safe; any external calls (e.g. to Perspective API) are opt-in and clearly namespaced.
6	Testable by default	Each guardrail ships with pytest fixtures + golden examples so devs can regression-test their own customisations.
7	Docs = Code-adjacent	Docstrings generate the API reference; narrative docs live in docs/ and build with MkDocs-Material.

3 High-level Architecture
swift
Copy
Edit
PsySafe AI SDK
 ├── Core
 │    ├── GuardrailBase   (abstract ↑↓)
 │    ├── PromptGuardrail (prompt-only)
 │    ├── CheckGuardrail  (post-generation validators)
 │    └── CompositeGuardrail
 │
 ├── Catalog (built-in guardrails)
 │    ├── vulnerability_detection/
 │    ├── complaints_handling/
 │    ├── pii_protection/
 │    └── mental_health_support/
 │
 ├── Integrations
 │    ├── openai_driver.py
 │    ├── anthropic_driver.py
 │    ├── transformers_driver.py
 │    └── langchain_adapter.py  (optional extra)
 │
 ├── Evaluation (place-holders)
 │    └── eval_runner.py
 │
 ├── CLI (optional)
 │    └── psysafe — “guard, test, report”
 │
 └── docs / examples / tests / etc.
4 Prompt-template Storage Strategy
After weighing YAML vs Python functions:

Hybrid, skewed to Python

Prompt content lives in .prompt.md files—pure, readable Markdown with special {{jinja}} slots for dynamic bits.

Prompt metadata & build logic live in one Python file per guardrail folder (guardrail.py).
This module defines a build() function returning a PromptTemplate Pydantic model that injects the .prompt.md text, default Jinja variables, and rendering logic.

Why?

Criterion	Markdown+Python	YAML
Core text editable by non-coders	✔ (Markdown)	✔
Strong type guarantees	✔ (Python models)	✖
IDE completion & reuse	✔	✖
Comfortable diff-review	✔ (one file for logic)	~
Multi-line prompt readability	✔	YAML scalar quirks

Developers can still ship fully-code guardrails (returning a string) if they want—PromptGuardrail.from_string("...")—but all built-ins follow the pattern:

bash
Copy
Edit
prompts/
  mental_health_support/
    prompt.md           # the human-friendly template
    guardrail.py        # the Pydantic builder + tests
    examples.jsonl      # golden IO pairs (used by pytest)
5 Public API Sketch (names may change)
python
Copy
Edit
from psysafe import GuardrailCatalog, OpenAIChatDriver

# 1) Pick a driver (or write your own)
driver = OpenAIChatDriver(model="gpt-4o-mini")

# 2) Compose guardrails
rail = GuardrailCatalog.load(
        ["pii_protection", "mental_health_support"]
      ).compose()        # returns CompositeGuardrail

# 3) Wrap the driver
safe_chat = rail.bind(driver)

# 4) Use as usual
resp = safe_chat("Hi, I'm feeling hopeless…")
print(resp.content)
All public classes are generics parametrised over the model’s native request / response types, so mypy can flag mismatches.

6 Core Components in Detail
Component	Responsibility	Key Types
PromptTemplate	Render .prompt.md + context into final string(s) for a given driver.	PromptTemplate, PromptRenderCtx
GuardrailBase	Abstract interface (apply(request) -> GuardedRequest and validate(response) -> ValidationReport).	BaseModel
PromptGuardrail	Implements apply() by pre-pending system messages. Validates nothing.	—
CheckGuardrail	Runs post-generation validators (regex, pydantic-based PII detection, etc.).	ValidationSeverity enum
CompositeGuardrail	Chains many guardrails; merges ValidationReports.	recursive composition
Driver Base	Uniform send/stream API plus minimal metadata (supports async).	ChatDriverABC
Catalog Loader	Discover guardrails on pkg_resources entry-points, so external libs can ship plugins.	GuardrailCatalog
CLI	psysafe guard <file> --with pii_protection	Click-based

7 Packaging & Tooling
Area	Decision
Build backend	pyproject.toml with Hatch (PEP 517)
Python versions	3.9 – 3.13 (typing-only features gated)
Runtime deps	Required: pydantic>=2, jinja2, httpx. Optional extras: openai, anthropic, transformers, langchain, perspective-api-client.
Lint / Format	Ruff + Black + MyPy strict
Testing	Pytest, coverage enforced ≥ 90 % for core
CI	GitHub Actions: matrix across py versions, lint, type-check, tests, docs build
Docs	MkDocs-Material + mike for versioned docs
Distribution	PyPI under name psysafe

8 Extensibility Points
Custom Guardrail

python
Copy
Edit
class AgeGateGuardrail(PromptGuardrail):
    """Refuse explicit content if user is < 18."""
GuardrailCatalog.register("age_gate", AgeGateGuardrail)
Third-party Driver

python
Copy
Edit
class MyLLMDriver(ChatDriverABC):
    ...
Validator Plug-in – provide a function (response) -> list[Violation], expose via entry-point group psysafe.validators.

9 Evaluation & Benchmarking Roadmap
Phase 1 (MVP)

Ship golden IO pairs (examples.jsonl) + pytest parametrised test that ensures no guardrail causes unsafe pass-through on its own examples.

Phase 2

Optional extras install psysafe[evaluate] pulls lightweight tox/harassment sets and Perspective-API scorer.

psysafe eval --model my.yaml --rail mental_health_support gives a Markdown report.

Phase 3

Integrate with OpenAI Evals or Helm via adapters.

10 Open Questions (for stakeholder feedback)
ID	Topic	Decision Needed
Q1	Crisis-line localisation in mental_health_support – embed numbers or let integrators inject?	
Q2	Ship strict default refusal templates vs. allow model-native style?	
Q3	Do we want a server-side audit log format (JSON Lines spec) from ValidationReports?	

11 Milestones & Timeline (tentative)
Week	Deliverable
1	Repo bootstrap, core abstractions, first two guardrails working on OpenAI driver.
2	Remaining built-in guardrails, Transformers driver, CLI alpha.
3	Docs MVP, pytest goldens, v0.1.0 tag on PyPI.
4+	Community feedback, add evaluation extras, first external plugin showcase.

12 Conclusion
This specification positions PsySafe AI as a developer-centric, drop-in safety layer that stays out of the way when you don’t need it and steps in forcefully when you do. With a hybrid prompt-template system, rigorous typing, and plug-in architecture, Python teams can adopt, extend, and trust PsySafe AI without fighting the tooling.

