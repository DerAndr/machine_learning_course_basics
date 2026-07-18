---
name: interactive-learning-experience-builder
description: Use when creating a grounded, accessible, self-contained interactive learning experience from repository knowledge sources.
---

# Interactive Learning Experience Builder

Create one deterministic `index.html` that opens through `file://` without a
server, network request, account, CDN, external font, or runtime dependency.
Embed the experience content, styles, scripts, SVG, fallbacks, and quiz banks.
Keep explanations and complete quiz content statically readable when JavaScript
or storage is unavailable.

## Workflow

1. Read repository instructions and any contributor guidance before choosing
   sources or editing files.
2. Follow [context discovery](references/context-discovery.md) to inventory
   authoritative knowledge sources, public-content boundaries, validation, and
   publishing conventions.
3. Write a short experience specification: learner, learning goals, named
   sources, concepts, visualizations, quiz intent, accessibility constraints,
   output location, and acceptance checks.
4. Read the [content contract](references/content-contract.md). Build a
   source-grounded JSON payload and correct unsupported claims, ambiguous
   answers, and missing provenance before generation.
5. Generate the HTML with `scripts/generate_learning_experience.py`,
   `assets/learning-experience-template.html`, and the embedded executable quiz
   state machine in `assets/quiz-state-machine.js`.
6. Validate the result with `scripts/validate_learning_experience.py`, then
   open the output directly through `file://` and exercise every setting,
   visualization fallback, quiz, keyboard path, visible focus state, reduced
   motion behavior, and storage-disabled behavior.
7. Run the repository's relevant checks and follow its publishing convention.

## Repository adapters

Start with this core workflow for one-off work. Create a thin repository adapter
only when the same local discovery, source-selection, validation, or publishing
workflow recurs. Base it on the
[repository adapter template](references/repository-adapter-template.md). Keep
stable local constraints in the adapter while leaving portable rendering,
accessibility, payload validation, and offline behavior in this core skill.

## Non-negotiable experience rules

- Use only named sources permitted by the repository context.
- Use repository-relative source paths, `http://` or `https://` provenance
  URLs, or explicit identifiers such as `kb:topic/item`.
- Do not add network-capable browser code or external runtime resources.
- Give every visualization a readable fallback that communicates the same
  essential lesson without JavaScript or color alone.
- Provide three quiz banks—`foundations`, `applied`, and `challenge`—with
  exactly 10 grounded questions each.
- Embed break-prompt content even when its initial display setting is off.
