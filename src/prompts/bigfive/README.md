# Big Five Prompts

Reserved directory for role-prior prompts under DyCo using Big Five as an alternative testbed. MBTI is only the current interpretable baseline.

## Quick migration
- Mirror the MBTI prompt layout under prompts/bigfive/<label>/.
- Keep file names aligned with BaseAgent calls: analyze_task.txt, evaluate_solution.txt, decide_team_preference.txt, attitude.txt, generate.txt, personality.txt.
- Update PromptLoader to read this taxonomy.

## Extensibility note
These prompts encode role priors; coordination (dynamic teaming and EVA) is shared.
