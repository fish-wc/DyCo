# HEXACO Agents

Reserved directory for DyCo role-prior agents based on HEXACO. The paper uses MBTI as an interpretable testbed only; this slot supports swapping the role-prior taxonomy while keeping coordination logic (dynamic teaming, willingness-guided speech, EVA).

## Quick migration
- Copy an MBTI agent from src/agents/mbti and map its type label to a HEXACO label.
- Register the new class in AGENT_CLASS_MAP (src/agents/agentsmanager.py).
- Route prompt loading to prompts/hexaco in PromptLoader and configs.

## Extensibility note
DyCo treats taxonomies as structured role priors, not personality claims. EVA and teaming remain unchanged.
