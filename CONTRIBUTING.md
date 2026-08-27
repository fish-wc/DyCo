# Contributing to DyCo

Thank you for your interest in DyCo. Contributions that improve reproducibility, evaluation reliability, documentation, and provider portability are welcome.

## Before opening an issue

- Search existing issues and pull requests first.
- For a bug, include the operating system, Python version, dependency installation method, relevant configuration (with secrets removed), and a minimal reproduction.
- For an evaluation report, include the benchmark version, model identifier, retrieval provider, task IDs, random seeds, and whether the result is based on automatic or human evaluation.
- Never include API keys, private prompts, private documents, or sensitive model outputs in an issue.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
```

Copy `.env.example` to `.env` only when a live API is needed. Keep `.env` local and use a small, non-sensitive test task while developing.

## Making changes

1. Create a focused branch from `main`.
2. Keep changes scoped: separate documentation, dependency, framework, and experiment changes when practical.
3. Preserve the distinction between role priors and claims about human personality. Do not describe MBTI prompts as psychological validation.
4. Do not commit generated workspaces, logs, model outputs, credentials, benchmark data, or compiled Python files.
5. Update the relevant README or configuration documentation when changing a command, parameter, output format, or external service.
6. For changes that affect reported numbers, state the model, endpoint, temperature, retrieval settings, task set, seeds, and token/cost budget.

## Local checks

Run these checks before submitting a pull request:

```bash
# Run the repository's offline tests
python -m pytest -q tests

# Parse all Python files
python -m compileall -q src configs evaluate tests

# Validate configuration files
python -c "import glob, yaml; [yaml.safe_load(open(p, encoding='utf-8')) for p in glob.glob('configs/**/*.yaml', recursive=True)]; print('YAML OK')"

# Check that the configuration package imports
python -c "from configs import config_loader; print(config_loader.get_agent_ids())"
```

If your change touches the live workflow, also run one small task with `--limit 1` and report whether external API calls were available. Do not treat a live API smoke test as a substitute for deterministic unit tests.

## Pull requests

A good pull request includes:

- a concise description of the problem and solution;
- the files or modules affected;
- validation commands and their results;
- any changes to dependencies or environment variables;
- reproducibility details for changed experiments;
- limitations or known follow-up work.

Please do not claim that a change improves benchmark performance without reporting the comparison protocol and uncertainty. Changes to the paper's numbers require author approval and a traceable evaluation artifact.

## Code style

Match the surrounding Python style and keep public interfaces documented. Prefer small, testable functions, explicit error messages, and configuration through environment variables rather than hard-coded secrets or provider-specific credentials.

## License

By contributing, you agree that your contribution may be distributed under the repository's [MIT License](LICENSE). Third-party datasets, APIs, and model outputs may have separate terms.
