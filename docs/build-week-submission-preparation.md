# Build Week Submission Preparation

This is a working worksheet, not final submission copy. Rewrite the public
description and tagline in your own voice before submitting. Keep only claims
you personally understand and can demonstrate.

The semantic companion revision is published on public `upstream/main` at
`a9fd6d9ec61f5c05981734846d858572413f5f92`. On 2026-07-20,
[Validate OKF run 29729743749](https://github.com/DerAndr/machine_learning_course_basics/actions/runs/29729743749)
and [Build Textbook Preview run 29729739505](https://github.com/DerAndr/machine_learning_course_basics/actions/runs/29729739505)
succeeded; the root and all three companion routes returned HTTP 200. The
remaining Devpost, video, screenshot, and submitter-authored work below is
still incomplete.

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
5. **The new insight was to generate short, semantic learning loops.** The
   project now provides a generic skill that turns repositories and knowledge
   bases into source-grounded companions whose controls change a
   topic-relevant interpretation. A thin repository-specific skill extends the
   generic workflow with local source, safety, and publishing rules.
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
2. **What it does:** turns named repository sources into short semantic
   interactive companions with explanations, topic-specific controls, three
   quiz levels, immediate feedback, accessibility settings, and offline
   fallbacks.
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
7. **Accomplishments:** one reusable workflow, one course adapter, and three
   companions whose interactions fit their topics: EDA explorations,
   Classification threshold/boundary reasoning, and Regression residual,
   regularization, and metric reasoning.
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
> without a server. Python 3.12 and `uv` are needed only for regeneration and
> automated validation; `uv` can provision and use the required Python 3.12
> runtime when it is not already installed.

Confirm the exact Codex surfaces and operating systems you personally tested
before placing that statement in the final form.

## Judge test flow

### Fast path: working product without setup

Target time: two minutes.

1. Open the textbook homepage.
2. Confirm three cards under **Open a fast review** and three links under
   **Fast reviews** in the sidebar.
3. Open Classification. Move the threshold and explain how the confusion
   matrix, precision, and recall change; then switch the decision boundary and
   confirm Class A/Class B labels and non-color cues remain meaningful.
4. Open Regression. Change the residual pattern, compare Ridge and Lasso
   regularization paths, and increase the adjustable error to show MAE/RMSE
   metric sensitivity.
5. Toggle the palette setting in both companions. Confirm the graph marks
   visibly change and that shape, pattern, labels, and position still carry the
   meaning.
6. Answer one quiz question incorrectly, read the feedback, change the answer,
   answer correctly, and move to the next question.
7. Switch to Challenge depth or toggle focus-friendly and color-blind-safe
   settings.
8. Briefly open EDA to confirm that the shared portable shell supports a
   different, topic-specific exploration set.

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
| 0:20–0:42 | Architecture: public source knowledge → generic core skill → ML-course adapter → semantic payload → short companion | Foreground the reusable core, then show how the adapter adds repository policy without owning the renderer. |
| 0:42–1:02 | One trusted lecture source beside its payload and generation/validation command | Explain how Codex and GPT-5.6 helped transform named sources while deterministic tooling keeps the result reviewable. |
| 1:02–1:32 | Classification threshold and decision-boundary explorers | Move both controls and narrate the changed confusion outcome and class-aware boundary. |
| 1:32–1:58 | Regression regularization and metric-sensitivity explorers | Compare Ridge/Lasso shrinkage and show why one large error affects RMSE more than MAE. |
| 1:58–2:18 | Palette toggle, then integration-evidence feedback section | Explain that learner review exposed repeated generic charts, ignored class meaning, and a perceptually weak palette; show the portable-core fix rather than hiding the defect. |
| 2:18–2:32 | Textbook homepage with EDA/Regression/Classification | Demonstrate regeneration, repeatability, and discovery. |
| 2:32–2:40 | Published SHA, successful Actions/Pages evidence, then the three companions | Show the verified public SHA and deployment evidence, then close with the broader use: quickly learning from any trusted repository or knowledge base. |

Required spoken points:

- what was built and who it helps;
- how Codex accelerated repository navigation, implementation, testing, and
  review;
- how GPT-5.6 contributed to architecture and grounded learning-content work;
- which decisions remained human;
- why this is a reusable system rather than three handcrafted pages;
- how learner-facing feedback led to an upstream portable-core improvement and
  regenerated outputs.

## Screenshot plan

Prepare five clean images from the verified public Pages routes:

1. **Discovery:** textbook homepage with all three review cards and the sidebar.
2. **Classification semantics:** threshold/confusion outcomes beside the
   class-aware decision boundary.
3. **Regression semantics:** regularization paths beside metric sensitivity;
   capture Ridge/Lasso and MAE/RMSE labels.
4. **Feedback:** the quiz immediately explaining an incorrect answer, followed
   by the correct state.
5. **Trust and repeatability:** public source → generic core → course adapter →
   payload → generator/validator flow, or final deterministic and `VALID`
   evidence.

Use short captions that explain why each screen matters; do not submit five
screens that all show the same page layout.

## Feedback story

Present the iteration as evidence that the workflow is inspectable and
improvable. A learner-facing review showed that Regression and Classification
looked too similar, that Classification class semantics were not reaching the
marks, and that the two palette modes were perceptually weak. Because the
generator is shared, the correction was made once in the portable core:
semantic schemas and pure models were added, labels were preserved, redundant
non-color cues were rendered, and both palettes became visibly distinct. The
course adapter and payloads were updated, then all three companions were
regenerated. This is stronger than hiding a defect: it demonstrates a real
feedback → diagnosis → reusable fix → verified-output loop.

## Final submission checklist

### Repository and deployment

- [x] Add the Codex/GPT-5.6 disclosure to the public README.
- [x] Publish the reviewed result to public `upstream/main` at `a9fd6d9ec61f5c05981734846d858572413f5f92`.
- [x] Verify the Pages workflow and all three live demo routes.
- [x] Confirm the public repository includes relevant licenses.

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
