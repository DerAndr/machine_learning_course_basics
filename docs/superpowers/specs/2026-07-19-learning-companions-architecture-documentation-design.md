# Learning Companions Architecture Documentation Design

## Goal

Explain the learning-companion concept and its implementation at a high
architectural level, then make the explanation easy to discover and use in both
the public student repository and the private teacher repository.

## Audience

The primary audience is maintainers and AI agents. A short opening section will
also explain the concept in student-facing language:

- a learning companion is a small interactive experience grounded in named
  course or knowledge-base sources;
- it complements lectures, textbooks, and practical work rather than replacing
  them;
- it guides a learner through explanation, exploration, knowledge checking, and
  corrective feedback.

The architecture sections may use repository and generation terminology, but
must define it in plain language.

## Documentation structure

Create one canonical shared document:

`docs/learning-companions-architecture.md`

Keep it identical in both repositories. It is the conceptual and architectural
reference. Keep `docs/interactive-lecture-learning-assistant.md` as the
operational guide containing commands and the ML-course workflow.

This separation prevents the architecture narrative from becoming a command
reference and prevents the operational guide from becoming an architectural
essay.

## Architecture narrative

The canonical document will describe these layers:

1. **Knowledge layer** — repository instructions, lectures, OKF concepts,
   notebooks, documentation, or another knowledge base.
2. **Context layer** — discovery of authoritative sources, learners, learning
   goals, safety boundaries, repository conventions, and acceptance checks.
3. **Skill layer** — the portable
   `interactive-learning-experience-builder` plus an optional thin repository
   adapter such as `ml-course-interactive-learning-assistant`.
4. **Content layer** — a short experience specification and grounded JSON
   payload with concepts, visualizations, quiz banks, break prompts, and named
   provenance.
5. **Runtime layer** — deterministic generator, HTML template, quiz state
   machine, accessibility behavior, and static fallbacks.
6. **Artifact layer** — one self-contained `index.html` that works through
   `file://` without accounts, servers, CDNs, fonts, or runtime network access.
7. **Assurance layer** — payload validation, offline-resource validation,
   executable quiz tests, repository policy checks, mobile Chrome checks, and
   deterministic regeneration.
8. **Delivery layer** — committed offline artifact, repository preview build,
   and optional static-site publishing.

Include a Mermaid flow diagram showing:

```text
knowledge sources
  -> context discovery and experience specification
  -> portable core + optional repository adapter
  -> grounded payload
  -> deterministic generation
  -> offline learning companion
  -> validation and browser checks
  -> local use and optional publication
```

The diagram must also show that validation can send the payload or artifact
back for correction before publication.

## Responsibility boundaries

Include a table with these ownership rules:

| Component | Owns | Must not own |
|---|---|---|
| Portable core skill | General workflow, content contract, generator, template, quiz state machine, accessibility, offline validation | ML-course paths, OKF rules, private/public course policy, a particular publishing platform |
| Repository adapter | Stable local source hierarchy, safety allowlist, paths, recurring checks, publishing convention | A copied generator, template, quiz engine, or portable accessibility rules |
| Experience specification and payload | Learner, goals, grounded explanations, controls, quiz content, provenance | Executable application logic or private sources |
| Generated companion | Embedded content and runtime needed by the learner | External dependencies, accounts, uploads, arbitrary execution, or hidden knowledge sources |
| Repository build and CI | Regression checks and optional publication | A second hand-maintained companion source |

## Portability model

Document two supported paths:

- **One-off experience:** inspect context, write an experience specification,
  create a payload, and use the portable core directly.
- **Recurring repository workflow:** create a thin adapter containing only
  stable repository-specific discovery, safety, path, and publishing rules.

Do not recommend one skill per lecture or topic. A lecture or topic gets an
experience specification and payload. An adapter is created only when a
repository-level workflow recurs.

Repositories without `AGENTS.md`, a build tool, or publishing configuration
remain supported. Stable repository paths, URLs, or knowledge-base identifiers
may provide provenance when the repository policy allows them.

## ML-course mapping

Explain the course-specific implementation without making it part of the
portable core:

- `lectures/` supplies full course explanations and provenance;
- `okf/` supplies durable concept metadata and relationships and remains
  read-only for standalone companion generation;
- the ML adapter accepts only canonical public course paths under the selected
  lecture and `okf/`;
- `lecture_experiences/content/<lecture_slug>.json` is the grounded payload;
- `lecture_experiences/<lecture_slug>/index.html` is the canonical offline
  artifact;
- the preview builder copies that artifact to
  `site/_build/demos/<lecture_slug>/index.html`;
- the public student repository publishes the Pages version;
- teacher-only sources, solutions, quiz banks, grading data, and unpublished
  material never enter a public companion.

The shared architecture document may describe the existence of the
teacher/student boundary but must not name or expose private source content.

## Usage sections

The canonical document will provide four short usage paths:

### Learner

Open the companion online or through `file://`, choose question depth and
accessibility preferences, explore explanations and visualizations, and use the
quiz as corrective practice.

### Maintainer

Choose sources, write the specification and payload, generate through the
appropriate adapter, validate, run tests, inspect the artifact, and publish
through the repository convention.

### AI agent

Read repository instructions, invoke the portable skill and applicable adapter,
name every source, obey safety boundaries, regenerate deterministically, and
report verification evidence.

### Unrelated repository

Use the portable skill directly for a one-off experience. Add an adapter only
after stable local rules recur.

## Cross-document updates

Update these shared files in both repositories:

- `README.md` — link the architecture document from the interactive lecture
  reviews section and distinguish concept from operational guide.
- `AGENTS.md` — add the architecture document to the interactive-review
  navigation order.
- `docs/interactive-lecture-learning-assistant.md` — label it as the operational
  guide and link to the architecture reference.
- `docs/contributing-to-textbook.md` — explain that learning companions
  complement OKF concepts and link to the architecture and operational guides.

The teacher repository may retain its existing private publishing documentation.
No teacher-only content will be added to the shared architecture document or
student repository.

## Verification

Extend `tests/test_interactive_learning_assistant_docs.py` to verify:

- the architecture document exists;
- both README and AGENTS navigation link to it;
- the operational and contribution guides link to it;
- the document names the portable core, repository adapter, experience
  specification, deterministic generation, self-contained artifact,
  validation, and student/teacher safety boundary;
- it contains the Mermaid architecture flow and responsibility table;
- it does not describe the portable core as ML-specific.

Run the focused documentation tests, full test suite, Markdown-sensitive
repository checks, strict OKF validation, and preview build in both repository
integrations.

## Non-goals

- No runtime, payload, quiz, template, or publishing behavior changes.
- No new learning companion is generated.
- No duplication of the operational command guide.
- No teacher-only paths or private source details are introduced into the
  student repository.
- No new repository adapter is created.
