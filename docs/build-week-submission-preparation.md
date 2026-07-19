# Build Week Submission Preparation

This is a working worksheet, not final submission copy. Rewrite the public
description and tagline in your own voice before submitting. Keep only claims
you personally understand and can demonstrate.

## Narrative spine

The strongest story is a progression, not a feature list:

1. **Long-form teaching did not fit students' attention patterns.** I teach
   classical machine learning at a university as a hobby. I first recorded
   complete YouTube lectures, but students rarely watched the long videos.
2. **Making the same knowledge readable was not enough.** I moved the material
   into a repository with Markdown notes and supporting text. Direct usage
   remained limited.
3. **Students revealed the useful interaction model.** They did use agents to
   navigate the repository and generate summaries. The valuable behavior was
   asking for a focused route through trusted material, not browsing another
   large content collection.
4. **A connected textbook improved structure, not attention.** Skills and a
   GitHub Pages learning book made the course easier to navigate, but students
   still rarely chose the book for quick review.
5. **The new insight was to generate short learning loops.** The project now
   provides a generic skill that turns repositories and knowledge bases into
   short, source-grounded learning companions. A thin repository-specific
   skill extends the generic workflow with local source, safety, and publishing
   rules.
6. **The ML course is proof, not the limit.** EDA, Regression, and
   Classification companions demonstrate explanation → exploration → quiz →
   immediate feedback. The same architecture can help anyone quickly learn the
   main ideas in another trusted knowledge base.

The central problem statement:

> People often have access to trustworthy knowledge but not enough time or
> attention to find the important ideas and practise them. Generic summaries
> are fast but can lose provenance, interaction, and reviewability.

The central solution statement:

> A reusable Codex skill converts trusted repositories into short, accessible,
> source-grounded learning companions, while an optional adapter preserves each
> repository's local rules.

## Devpost field worksheet

### Core project fields

| Field | Working content |
|---|---|
| Project name | Choose this personally; do not outsource the final name. |
| Tagline starter | A reusable Codex skill that turns trusted repositories into short, interactive learning companions. Rewrite this in your own voice. |
| Category | **Education** |
| Repository | <https://github.com/DerAndr/machine_learning_course_basics> |
| Built with | Codex, GPT-5.6, Python, HTML, CSS, JavaScript, pytest, GitHub Pages, Open Knowledge Format |
| Public demo | <https://derandr.github.io/machine_learning_course_basics/> |
| Video | Public YouTube URL; under three minutes; spoken explanation required |
| `/feedback` | Paste the Session ID from the Codex task containing most of the core development |

### Project-description outline

Use these as factual prompts, then rewrite them as natural prose in your own
voice:

1. **Inspiration:** teaching classical ML; long YouTube lectures; limited
   adoption of notes and the textbook; students successfully using agents for
   navigation and summaries.
2. **What it does:** turns named repository sources into short interactive
   reviews with explanations, visual controls, three quiz levels, immediate
   feedback, accessibility settings, and offline fallbacks.
3. **How it works:** a portable core skill owns the content contract,
   deterministic generator, quiz state machine, and validator; a thin
   ML-course adapter owns repository source and publishing policy; each topic
   owns only grounded JSON and generated HTML.
4. **How Codex and GPT-5.6 were used:** repository discovery, architecture,
   bounded implementation tasks, source-grounded content synthesis, tests,
   deterministic regeneration, and browser review.
5. **Human decisions:** learner problem, source authority, public/private
   boundaries, accessibility defaults, pedagogical sequence, and final review.
6. **Challenges:** keeping generated questions grounded and unambiguous;
   preserving offline portability; preventing adapters from duplicating the
   core; making interaction accessible; proving generation is deterministic.
7. **Accomplishments:** one reusable workflow, one course adapter, three
   working companions, textbook discovery, 30 questions per topic, offline
   validation, and automated plus desktop/mobile verification.
8. **What comes next:** test the workflow with other courses and knowledge
   bases, gather learner feedback, and improve the short learning loop without
   weakening source provenance.

### Live and offline demonstrations

| Topic | Live route | Offline artifact |
|---|---|---|
| Exploratory Data Analysis | <https://derandr.github.io/machine_learning_course_basics/demos/lecture_01_eda/> | `lecture_experiences/lecture_01_eda/index.html` |
| Regression | <https://derandr.github.io/machine_learning_course_basics/demos/lecture_04_regression/> | `lecture_experiences/lecture_04_regression/index.html` |
| Classification Part 1 | <https://derandr.github.io/machine_learning_course_basics/demos/lecture_05_classification_part_1/> | `lecture_experiences/lecture_05_classification_part_1/index.html` |

### Plugin or developer-tool field

Although the project is entered in Education, the reusable skill is also a
developer-facing workflow. A concise judge instruction can say:

> Clone the repository and open its root in Codex so the repository-local
> `.agents/skills/` workflows are discoverable. Use the general-purpose prompt
> from `docs/interactive-lecture-learning-assistant.md`, or use the ML-course
> prompt for a lecture-scoped review. Generated HTML works in a modern browser
> without a server. Python 3.11+ and `uv` are needed only for regeneration and
> automated validation.

Confirm the exact Codex surfaces and operating systems you personally tested
before placing that statement in the final form.

## Judge test flow

### Fast path: working product without setup

Target time: two minutes.

1. Open the textbook homepage.
2. Confirm three cards under **Open a fast review** and three links under
   **Fast reviews** in the sidebar.
3. Open Regression.
4. Change a visualization control and confirm the nearby fallback still
   explains the same lesson.
5. Answer one quiz question incorrectly, read the feedback, change the answer,
   answer correctly, and move to the next question.
6. Switch to Challenge depth or toggle focus-friendly and color-blind-safe
   settings.
7. Briefly open EDA and Classification to confirm the same reusable structure
   with different grounded content.

### Offline path

1. Clone the repository.
2. Open any
   `lecture_experiences/<lecture_slug>/index.html` directly in a browser.
3. Confirm that concepts, visualizations, quiz feedback, and static answer
   review work without a server or network request.

### Reproducibility path

```bash
uv sync
uv run pytest
uv run python tools/validate_okf.py okf/ --strict-warnings
uv run python tools/build_textbook_preview.py
uv run python .agents/skills/interactive-learning-experience-builder/scripts/validate_learning_experience.py lecture_experiences/lecture_04_regression/index.html
```

Expected evidence:

- the full test suite passes;
- strict OKF validation reports no diagnostics;
- the preview publishes all matching payload/artifact pairs;
- the offline validator reports `VALID`;
- topic tests regenerate committed artifacts byte-for-byte.

### Skill path

From the repository root in Codex, use one of these prompts:

> Use $interactive-learning-experience-builder to create a grounded, offline
> interactive learning experience from this repository's knowledge sources.

> Use $ml-course-interactive-learning-assistant with
> $interactive-learning-experience-builder to create a grounded, accessible,
> self-contained review for a selected ML-course lecture.

For judging, demonstrate the workflow with an existing committed payload rather
than spending demo time waiting for a new generation.

## Demo video flow

Target length: 2 minutes 40 seconds. Record a clear public YouTube video under
three minutes with continuous spoken explanation.

| Time | Screen | Spoken purpose |
|---|---|---|
| 0:00–0:20 | Brief view of the course repository or old lecture list | Explain the progression from long lectures, to notes, to agent navigation, and the unmet attention problem. |
| 0:20–0:40 | Small architecture view: sources → generic skill → optional adapter → short companion | State the reusable idea and why source grounding and repository policy are separate. |
| 0:40–1:05 | Show one lecture source, its JSON payload, and the generation/validation command | Explain how Codex and GPT-5.6 helped transform trusted sources while deterministic tooling keeps the result reviewable. |
| 1:05–1:50 | Regression companion | Show one concept, one graph control and fallback, then wrong answer → feedback → corrected answer → next question. |
| 1:50–2:10 | Textbook homepage and the EDA/Classification cards | Demonstrate repeatability and discovery rather than teaching every topic. |
| 2:10–2:28 | Test output or integration evidence | Show source-policy checks, 30 questions per topic, offline validation, deterministic regeneration, and mobile review. |
| 2:28–2:40 | Return to the three companions | Close with the broader use: quickly learning from any trusted repository or knowledge base. |

Required spoken points:

- what was built and who it helps;
- how Codex accelerated repository navigation, implementation, testing, and
  review;
- how GPT-5.6 contributed to architecture and grounded learning-content work;
- which decisions remained human;
- why this is a reusable system rather than three handcrafted pages.

## Screenshot plan

Prepare four clean images after the Pages deployment is verified:

1. **Discovery:** textbook homepage with all three review cards and the sidebar.
2. **Learning loop:** one companion showing a concise concept beside an
   interactive visualization and text fallback.
3. **Feedback:** the quiz immediately explaining an incorrect answer, followed
   by the correct state.
4. **Trust and repeatability:** source/payload/generator/validator flow or a
   concise terminal view showing deterministic tests and `VALID` results.

Use short captions that explain why each screen matters; do not submit four
screens that all show the same page layout.

## Final submission checklist

### Repository and deployment

- [ ] Add the Codex/GPT-5.6 disclosure to the public README.
- [ ] Merge `codex/build-week-showcase` into `basics`.
- [ ] Push the reviewed result to `upstream/main`.
- [ ] Verify the Pages workflow and all three live demo routes.
- [ ] Confirm the public repository includes relevant licenses.

### Devpost project

- [ ] Choose the project name personally.
- [ ] Rewrite the tagline and description in your own voice.
- [ ] Select submitter type and country of residence.
- [ ] Select **Education**.
- [ ] Add the repository and optional live test URL.
- [ ] Add installation and judge-testing instructions for the reusable skill.
- [ ] Add all team members and confirm they accepted, if applicable.

### Required evidence

- [ ] Run `/feedback` in the primary Codex build task and save its Session ID.
- [ ] Capture final screenshots after deployment.
- [ ] Record and upload the public YouTube video under three minutes.
- [ ] Confirm the audio names both Codex and GPT-5.6 and explains their use.
- [ ] Re-run the judge test flow from a clean checkout.
- [ ] Review the current official rules and announcements.
- [ ] Submit before July 21, 2026 at 5:00 PM Pacific Time.
