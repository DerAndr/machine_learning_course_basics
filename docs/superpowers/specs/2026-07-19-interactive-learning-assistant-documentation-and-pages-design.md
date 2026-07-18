# Interactive Learning Assistant Documentation and Pages Design

## Goal

Document the interactive lecture learning assistant and its Lecture 01 EDA
example in both course repositories, publish the example through the student
repository's existing GitHub Pages build, and merge the verified work into both
`main` branches without removing teacher-only material.

## Repository roles

The student repository, `machine_learning_course_basics`, is the public source
for the skill, content payload, standalone demo, usage documentation, and live
GitHub Pages experience.

The teacher repository, `machine_learning_course_teacher`, receives the same
public-safe feature and documentation while preserving its private authoring
history and teacher-only files. Its publishing documentation also explains how
interactive lecture experiences flow safely to the student repository.

## Student-facing documentation

Both repositories will share these public documentation changes:

- `README.md` introduces interactive lecture reviews and links to the usage
  guide, offline EDA file, and live Pages demo.
- `docs/interactive-lecture-learning-assistant.md` explains learner controls,
  offline use, the EDA example, the repository-local skill, payload generation,
  validation, public-source restrictions, and extension to other lectures.
- `lectures/lecture_01_eda/README.md` links the EDA lecture to its interactive
  review.
- `AGENTS.md` documents the skill, `lecture_experiences/content/`, generated
  experiences, navigation order, and required validation commands.

The live URL is:

```text
https://derandr.github.io/machine_learning_course_basics/demos/lecture_01_eda/
```

The offline source remains:

```text
lecture_experiences/lecture_01_eda/index.html
```

## Pages publication architecture

`lecture_experiences/` remains the source of standalone lecture-review
experiences. `tools/build_textbook_preview.py` copies each generated
`lecture_experiences/<lecture_slug>/index.html` into:

```text
site/_build/demos/<lecture_slug>/index.html
```

The builder copies the existing self-contained HTML byte-for-byte. It does not
maintain a second site template or duplicate generated source under `site/`.
This keeps the downloadable offline file and live Pages page synchronized.

The initial implementation publishes only directories containing an
`index.html`. Content JSON remains repository source and is not copied into the
Pages artifact.

## Workflow integration

The GitHub Pages workflow watches:

- `lecture_experiences/**`
- the learning-assistant skill
- the builder and its regression tests
- the new documentation paths

Local preview builds must create the EDA demo at the same relative route used
by Pages. Regression tests compare the published demo bytes with the committed
standalone HTML and confirm the documented route exists.

## Teacher-only documentation

`docs/publishing-model.md` changes only on the teacher integration branch. It
records that:

- interactive lecture payloads and generated HTML must use public lecture or
  read-only OKF sources;
- private solutions, grading artifacts, quiz banks, and teacher notes must not
  enter the payload or generated page;
- the student repository is the canonical public distribution point;
- the standalone file and Pages copy must be generated from the same verified
  source.

## Merge strategy

1. Implement shared documentation, Pages publication, workflow triggers, and
   tests on `codex/interactive-learning-assistant`.
2. Fetch both remotes immediately before integration.
3. Merge the feature branch into the student `main` branch and run the complete
   student validation suite on the merged result.
4. Push student `main`, verify GitHub Actions, then require the live demo URL to
   return HTTP 200 and contain the EDA review heading.
5. Merge the same feature branch into a teacher integration branch based on
   `origin/main`.
6. Add and commit the teacher-only publishing-model update.
7. Run the complete suite on the teacher merged result, merge into teacher
   `main`, and push `origin/main`.

No force-pushes, history rewrites, or deletion of teacher-only files are
allowed. If either remote `main` moves during integration, fetch and incorporate
the new tip before pushing.

## Validation

Shared feature validation includes:

- all repository tests;
- Ruff formatting and lint checks;
- strict OKF validation and textbook preview generation;
- learning-assistant skill validation;
- standalone HTML validation;
- a regression test proving the Pages artifact contains the exact committed
  EDA HTML;
- browser checks for the local Pages route;
- confirmation that the shared feature does not alter `okf/`.

After the student push:

- the Pages workflow must succeed;
- the live EDA URL must return HTTP 200;
- the page must contain `Exploratory Data Analysis: Interactive Review`;
- no external runtime resources may be required.

After the teacher push:

- the teacher repository's required checks must succeed;
- teacher-only publishing documentation and private assets must remain present.

## Failure handling

- A failing local check blocks both merges.
- A student Pages failure blocks the teacher push until the shared publication
  issue is fixed or clearly isolated to repository configuration.
- A merge conflict is resolved against each repository's role: public-safe
  common documentation in both, teacher-only material only in the teacher
  repository.
- A persistent live 404 after a successful build triggers inspection of the
  Pages deployment job and artifact route before any success claim.
