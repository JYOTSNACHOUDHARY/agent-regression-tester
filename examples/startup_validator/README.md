# Example agent: Startup Idea Validator

The agent under test in this repo's demo. It's a copy of
[JYOTSNACHOUDHARY/startupIdeaGenerator](https://github.com/JYOTSNACHOUDHARY/startupIdeaGenerator),
a LangGraph workflow that generates startup ideas for a domain, researches them with
five tools, and gives a Go / No-Go verdict for each.

Changes from the original:

| Change | Why |
|---|---|
| `langchain.prompts` → `langchain_core.prompts` | LangChain 1.x moved prompt templates |
| `risk_score_calculator` seeded from the idea text | Recorded runs are reproducible |
| Validator prompt requires every tool | **Found by agentreg:** about 10% of runs skipped a research tool (4 of 40 runs before the fix, 0 of 20 after) |
| `decision_maker` reads every tool result | Bug: it only read the *last* tool message, so its verdicts were based on a single risk score |
