# Frontier Judge Task Specification

This document is fed verbatim to the frontier judge (Claude/GPT/Gemini/
OpenRouter, whichever the user configures) as part of its system prompt,
for both **generating** tasks for the local model and **grading** its
responses. It is the anti-hallucination contract for this phase of
localbench: unlike the frontier judge's own general knowledge, everything
it does here must be traceable to something concrete.

## The core rule

Task content must either be:

1. **Fully self-contained in the prompt** — everything needed to complete
   and grade the task is given directly, so no outside knowledge is
   required from either the local model or the judge, or
2. **Programmatically verifiable** — the correct answer can be computed by
   the harness (arithmetic, code execution, schema validation), not
   recalled from the judge's training data.

**The judge is never asked to verify external real-world facts from memory
as ground truth.** A frontier model can be confidently wrong about facts
too; using its recall as an oracle would just move the hallucination risk
from the model under test to the judge. The judge's role is restricted to
qualitative axes — clarity, coherence, constraint-following, internal
consistency — layered on top of ground truth that's either handed to it in
the prompt or computed by the harness.

## Categories

### instruction_following
Multi-constraint instructions (format, length, ordering, negative
constraints like "don't use word X"). Self-grounded: constraints are
checkable directly against the response text, no outside knowledge needed.

### source_grounded_summarization
The judge supplies the source passage **directly in the prompt** ("here is
a passage: ...") and asks the local model to summarize or answer questions
about *only* that passage. Because the source is given in-context, no
external factual recall is required of either party — grading checks
whether the summary is faithful to the *given* text, not to the judge's
background knowledge of the topic.

### reasoning_explanation
Multi-step logic/reasoning problems where the judge derives the correct
answer from the problem's own stated constraints (not from memory) before
grading. The judge checks whether the local model's steps are valid and
the conclusion matches the judge's own from-scratch derivation.

### creative_writing
Judged only on constraint-adherence (genre, length, tone, required
elements) and internal consistency — explicitly **not** a fact-checking
category. Subjective creative quality is out of scope for scoring.

### code_explanation
The judge is given a self-contained code snippet and, wherever practical,
executes it (the same way the deterministic `coding` suite does) to obtain
actual ground-truth behavior *before* grading — so the judge verifies the
local model's explanation against real execution output, not against its
own guess at what the code does.

### hallucination_probe (the deliberate exception)
The one category that intentionally tests for confident fabrication. The
harness constructs a plausible-but-fictitious premise itself (e.g. a
function/library/person name that sounds real but is invented at
generation time, so its non-existence is certain by construction, not by
the judge's uncertain recall) and checks whether the local model asserts
confident false detail or appropriately hedges/declines. This is the only
category where "the premise is false" is asserted rather than
self-evident from the prompt — and it's true by construction, not by
anyone's memory.

## Grading output format

The judge must always respond with structured JSON, never a free-form
verdict:

```json
{
  "score": 0-10,
  "constraints_met": ["list of which specific constraints/criteria were satisfied"],
  "issues": ["list of specific problems found, if any"],
  "rationale": "one or two sentences explaining the score"
}
```

Structured, itemized output is used deliberately to mitigate two documented
LLM-judge biases: **verbosity bias** (judges tend to prefer longer answers)
and **position/framing bias** — decomposing the rubric into atomic,
independently-checkable criteria keeps the score anchored to specifics
rather than a vague overall impression.

## Standing cautions for whoever configures the judge

- **Avoid same-family self-preference bias**: don't judge a model with a
  judge from the same base model family (e.g. don't grade a Llama-derived
  local model with a Llama-derived judge via OpenRouter) where avoidable.
- **This is directional signal, not a certified benchmark**: proper
  human-calibration of an LLM-judge typically wants several hundred
  labeled examples for low variance — not practical for a personal local-
  model bench. Treat scores as a useful comparative signal between models
  you test, not an absolute, portable quality metric.
