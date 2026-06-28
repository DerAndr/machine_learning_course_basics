"""Deterministic validation for the course Open Knowledge Format bundle."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

import yaml

RESERVED_FILES = {"index.md", "log.md"}
REQUIRED_FIELDS = ("type", "title", "description", "tags", "timestamp", "status")
RELATIONSHIP_FIELDS = ("prerequisites", "related_concepts", "related_labs")
VALID_STATUSES = {"draft", "review", "published", "deprecated"}
VALID_DIFFICULTIES = {"introductory", "intermediate", "advanced"}
CONTROLLED_TYPES = {
    "Concept",
    "Course",
    "Dataset",
    "Exercise",
    "Glossary Term",
    "Interactive Lab",
    "Learning Path",
    "Metric",
    "ML Algorithm",
    "Module",
    "Pitfall",
    "Reference",
    "Worked Example",
}
VALID_TAGS = {
    "algorithm",
    "classification",
    "clustering",
    "data-preparation",
    "dataset",
    "evaluation",
    "foundations",
    "interactive",
    "interpretability",
    "metric",
    "ml-systems",
    "pitfall",
    "regression",
    "responsible-ai",
    "supervised-learning",
    "unsupervised-learning",
}
NON_INSTRUCTIONAL_TYPES = {"Dataset", "Glossary Term", "Reference"}
FORBIDDEN_SOURCE_PARTS = (
    "_teacher_90min.ipynb",
    "teacher_cheat_sheet.md",
    "/quizzes/",
    "/answer_keys/",
    "/incoming_materials/",
    "/legacy_import/",
)
FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*(?:\n|\Z)", re.DOTALL)
MARKDOWN_LINK_RE = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")
INDEX_ENTRY_RE = re.compile(r"^\s*[*-]\s+\[[^\]]+\]\(([^)]+)\)\s+-\s+(\S.*)\s*$")
HEADING_RE = re.compile(r"^(#{1,6})\s+\S", re.MULTILINE)
LOG_HEADING_RE = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)
PATH_PART_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
TAG_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
FENCED_CODE_RE = re.compile(r"(^|\n)(`{3,}|~{3,})[^\n]*\n.*?\n\2[ \t]*(?=\n|$)", re.DOTALL)


@dataclass(frozen=True, order=True)
class Diagnostic:
    """One stable validator finding."""

    path: str
    line: int
    code: str
    severity: str
    message: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return asdict(self)


@dataclass(frozen=True)
class ValidationResult:
    """The complete result of validating one bundle."""

    diagnostics: tuple[Diagnostic, ...]
    concept_count: int
    index_count: int

    @property
    def errors(self) -> tuple[Diagnostic, ...]:
        return tuple(item for item in self.diagnostics if item.severity == "error")

    @property
    def warnings(self) -> tuple[Diagnostic, ...]:
        return tuple(item for item in self.diagnostics if item.severity == "warning")


@dataclass(frozen=True)
class ParsedMarkdown:
    metadata: dict[str, Any] | None
    body: str
    frontmatter_error: str | None = None


def parse_markdown(text: str) -> ParsedMarkdown:
    """Split YAML frontmatter from a Markdown body."""

    normalized = text.replace("\r\n", "\n")
    match = FRONTMATTER_RE.match(normalized)
    if not match:
        return ParsedMarkdown(metadata=None, body=normalized)
    try:
        loaded = yaml.safe_load(match.group(1))
    except yaml.YAMLError as exc:
        return ParsedMarkdown(
            metadata=None, body=normalized[match.end() :], frontmatter_error=str(exc)
        )
    if loaded is None:
        loaded = {}
    if not isinstance(loaded, dict):
        return ParsedMarkdown(
            metadata=None,
            body=normalized[match.end() :],
            frontmatter_error="frontmatter must be a YAML mapping",
        )
    return ParsedMarkdown(metadata=loaded, body=normalized[match.end() :])


def _display_path(path: Path, bundle: Path) -> str:
    return path.relative_to(bundle).as_posix()


def _add(
    diagnostics: list[Diagnostic],
    path: str,
    code: str,
    severity: str,
    message: str,
    line: int = 1,
) -> None:
    diagnostics.append(Diagnostic(path, line, code, severity, message))


def _is_url(target: str) -> bool:
    return urlsplit(target).scheme in {"http", "https", "mailto"}


def _resolve_local_link(source: Path, target: str, bundle: Path) -> Path | None:
    clean = unquote(target.split("#", 1)[0].split("?", 1)[0])
    if not clean or _is_url(clean):
        return None
    candidate = bundle / clean.lstrip("/") if clean.startswith("/") else source.parent / clean
    if clean.endswith("/") or candidate.is_dir():
        candidate /= "index.md"
    return candidate.resolve()


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _strip_fenced_code(body: str) -> str:
    return FENCED_CODE_RE.sub("\n", body)


def _valid_timestamp(value: Any) -> bool:
    if isinstance(value, datetime):
        return value.tzinfo is not None
    if not isinstance(value, str) or not value.strip():
        return False
    if "T" not in value:
        return False
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return parsed.tzinfo is not None


def _validate_path_shape(path: Path, bundle: Path, diagnostics: list[Diagnostic]) -> None:
    relative = path.relative_to(bundle)
    for part in relative.parts[:-1]:
        if not PATH_PART_RE.fullmatch(part):
            _add(
                diagnostics,
                relative.as_posix(),
                "OKF060",
                "error",
                f"directory name must be lowercase kebab-case: {part}",
            )
    if path.name not in RESERVED_FILES:
        stem = path.stem
        if not PATH_PART_RE.fullmatch(stem):
            _add(
                diagnostics,
                relative.as_posix(),
                "OKF061",
                "error",
                f"concept filename must be lowercase kebab-case: {path.name}",
            )


def _validate_headings(body: str, path: str, diagnostics: list[Diagnostic]) -> None:
    levels = [len(match.group(1)) for match in HEADING_RE.finditer(body)]
    if not levels:
        _add(diagnostics, path, "OKF030", "warning", "document has no Markdown heading")
        return
    if levels.count(1) > 1:
        _add(diagnostics, path, "OKF031", "warning", "document has more than one H1 heading")
    for previous, current in zip(levels, levels[1:], strict=False):
        if current > previous + 1:
            _add(diagnostics, path, "OKF032", "warning", "heading levels skip a level")
            break


def _validate_target(
    source: Path,
    target: str,
    bundle: Path,
    display: str,
    diagnostics: list[Diagnostic],
    *,
    leading_slash_warning: bool = False,
) -> Path | None:
    if _is_url(target) or target.startswith("#"):
        return None
    if leading_slash_warning and target.startswith("/"):
        _add(
            diagnostics,
            display,
            "OKF020",
            "warning",
            "body and index links should be relative, not leading-slash paths",
        )
    resolved = _resolve_local_link(source, target, bundle)
    if resolved is not None:
        if not _is_relative_to(resolved, bundle):
            _add(diagnostics, display, "OKF023", "error", f"local link escapes bundle: {target}")
            return None
        if not resolved.exists():
            _add(
                diagnostics,
                display,
                "OKF021",
                "error",
                f"local link target does not exist: {target}",
            )
    return resolved


def _validate_index(
    path: Path,
    parsed: ParsedMarkdown,
    bundle: Path,
    diagnostics: list[Diagnostic],
) -> set[Path]:
    display = _display_path(path, bundle)
    linked: set[Path] = set()
    if parsed.frontmatter_error:
        _add(diagnostics, display, "OKF003", "error", parsed.frontmatter_error)
    if path == bundle / "index.md":
        if parsed.metadata is not None and set(parsed.metadata) != {"okf_version"}:
            _add(
                diagnostics,
                display,
                "OKF010",
                "error",
                "root index frontmatter may contain only okf_version",
            )
        if parsed.metadata is not None and str(parsed.metadata.get("okf_version")) != "0.1":
            _add(diagnostics, display, "OKF011", "error", 'root okf_version must be "0.1"')
    elif parsed.metadata is not None:
        _add(diagnostics, display, "OKF012", "error", "nested index files must not use frontmatter")

    for line_number, line in enumerate(parsed.body.splitlines(), 1):
        if not re.match(r"^\s*[*-]\s+\[", line):
            continue
        match = INDEX_ENTRY_RE.match(line)
        if not match:
            _add(
                diagnostics,
                display,
                "OKF013",
                "error",
                "index entries must use '[Title](target) - description'",
                line_number,
            )
            continue
        target, description = match.groups()
        if not description.strip():
            _add(diagnostics, display, "OKF014", "error", "index description is empty", line_number)
        resolved = _validate_target(
            path, target, bundle, display, diagnostics, leading_slash_warning=True
        )
        if resolved is not None and resolved.exists():
            linked.add(resolved)
    _validate_headings(parsed.body, display, diagnostics)
    return linked


def _validate_concept(
    path: Path,
    parsed: ParsedMarkdown,
    bundle: Path,
    repository_root: Path,
    diagnostics: list[Diagnostic],
) -> set[Path]:
    display = _display_path(path, bundle)
    linked: set[Path] = set()
    if parsed.frontmatter_error:
        _add(diagnostics, display, "OKF003", "error", parsed.frontmatter_error)
        return linked
    if parsed.metadata is None:
        _add(diagnostics, display, "OKF001", "error", "concept is missing YAML frontmatter")
        return linked
    metadata = parsed.metadata
    for field in REQUIRED_FIELDS:
        value = metadata.get(field)
        if value is None or value == "" or value == []:
            _add(
                diagnostics,
                display,
                "OKF002",
                "error",
                f"required field is missing or empty: {field}",
            )

    for field in ("title", "description"):
        value = metadata.get(field)
        if value is not None and (not isinstance(value, str) or not value.strip()):
            _add(diagnostics, display, "OKF024", "error", f"{field} must be a non-empty string")

    concept_type = metadata.get("type")
    if concept_type is not None and (not isinstance(concept_type, str) or not concept_type.strip()):
        _add(diagnostics, display, "OKF004", "error", "type must be a non-empty string")
    elif isinstance(concept_type, str) and concept_type not in CONTROLLED_TYPES:
        _add(
            diagnostics,
            display,
            "OKF040",
            "warning",
            f"type is outside the local vocabulary: {concept_type}",
        )

    status = metadata.get("status")
    if status is not None and (not isinstance(status, str) or status not in VALID_STATUSES):
        _add(diagnostics, display, "OKF005", "error", f"unsupported status: {status}")
    tags = metadata.get("tags")
    if tags is not None and (
        not isinstance(tags, list)
        or not tags
        or any(not isinstance(tag, str) or not TAG_RE.fullmatch(tag) for tag in tags)
    ):
        _add(
            diagnostics,
            display,
            "OKF006",
            "error",
            "tags must be a non-empty list of kebab-case strings",
        )
    elif isinstance(tags, list):
        for tag in tags:
            if tag not in VALID_TAGS:
                _add(
                    diagnostics,
                    display,
                    "OKF043",
                    "warning",
                    f"tag is outside the local vocabulary: {tag}",
                )
    timestamp = metadata.get("timestamp")
    if timestamp is not None and not _valid_timestamp(timestamp):
        _add(diagnostics, display, "OKF007", "error", "timestamp must be an ISO 8601 UTC time")

    difficulty = metadata.get("difficulty")
    if difficulty is not None and (
        not isinstance(difficulty, str) or difficulty not in VALID_DIFFICULTIES
    ):
        _add(diagnostics, display, "OKF025", "error", "difficulty uses an unsupported value")

    reading_minutes = metadata.get("estimated_reading_minutes")
    if reading_minutes is not None and (
        not isinstance(reading_minutes, int) or not 1 <= reading_minutes <= 30
    ):
        _add(
            diagnostics,
            display,
            "OKF026",
            "error",
            "estimated_reading_minutes must be an integer from 1 to 30",
        )

    objectives = metadata.get("learning_objectives")
    is_instructional = (
        not isinstance(concept_type, str) or concept_type not in NON_INSTRUCTIONAL_TYPES
    )
    if is_instructional and (
        not isinstance(objectives, list)
        or not 1 <= len(objectives) <= 3
        or any(not isinstance(item, str) or not item.strip() for item in objectives)
    ):
        _add(
            diagnostics,
            display,
            "OKF008",
            "error",
            "instructional concepts require one to three learning_objectives",
        )

    for field in RELATIONSHIP_FIELDS:
        values = metadata.get(field, [])
        if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
            _add(diagnostics, display, "OKF015", "error", f"{field} must be a list of paths")
            continue
        for value in values:
            if not value.startswith("/"):
                _add(
                    diagnostics,
                    display,
                    "OKF016",
                    "error",
                    f"{field} path must begin with '/': {value}",
                )
                continue
            resolved = _validate_target(path, value, bundle, display, diagnostics)
            if resolved is not None and resolved.exists():
                linked.add(resolved)

    sources = metadata.get("source_materials", [])
    if sources and (
        not isinstance(sources, list) or any(not isinstance(item, str) for item in sources)
    ):
        _add(
            diagnostics,
            display,
            "OKF017",
            "error",
            "source_materials must be a list of paths or URLs",
        )
    elif isinstance(sources, list):
        for source in sources:
            normalized = "/" + source.lower().replace("\\", "/").lstrip("/")
            if any(part in normalized for part in FORBIDDEN_SOURCE_PARTS):
                _add(
                    diagnostics,
                    display,
                    "OKF018",
                    "error",
                    f"teacher-only source is forbidden: {source}",
                )
            elif (
                source.startswith("/lectures/")
                and not (repository_root / source.lstrip("/")).exists()
            ):
                _add(
                    diagnostics,
                    display,
                    "OKF019",
                    "error",
                    f"repository source does not exist: {source}",
                )
            elif not source.startswith("/lectures/") and not _is_url(source):
                _add(
                    diagnostics,
                    display,
                    "OKF022",
                    "error",
                    f"source_materials must use /lectures/ paths or external URLs: {source}",
                )

    for target in MARKDOWN_LINK_RE.findall(_strip_fenced_code(parsed.body)):
        resolved = _validate_target(
            path, target, bundle, display, diagnostics, leading_slash_warning=True
        )
        if resolved is not None and resolved.exists():
            linked.add(resolved)
    if metadata.get("citation_required") is True and "# Citations" not in parsed.body:
        _add(
            diagnostics,
            display,
            "OKF041",
            "warning",
            "citation_required is true but # Citations is absent",
        )
    _validate_headings(parsed.body, display, diagnostics)
    return linked


def _validate_log(
    path: Path, parsed: ParsedMarkdown, bundle: Path, diagnostics: list[Diagnostic]
) -> None:
    display = _display_path(path, bundle)
    if parsed.metadata is not None or parsed.frontmatter_error:
        _add(diagnostics, display, "OKF050", "error", "log files must not use frontmatter")
    headings = LOG_HEADING_RE.findall(parsed.body)
    dates: list[str] = []
    for heading in headings:
        try:
            datetime.strptime(heading, "%Y-%m-%d")
        except ValueError:
            _add(
                diagnostics,
                display,
                "OKF051",
                "error",
                f"log date heading must use YYYY-MM-DD: {heading}",
            )
            continue
        dates.append(heading)
    if dates != sorted(dates, reverse=True):
        _add(diagnostics, display, "OKF052", "error", "log date headings must be newest first")
    _validate_headings(parsed.body, display, diagnostics)


def validate_bundle(bundle_path: Path, repository_root: Path | None = None) -> ValidationResult:
    """Validate an OKF bundle and return sorted diagnostics."""

    bundle = bundle_path.resolve()
    repo = (repository_root or bundle.parent).resolve()
    diagnostics: list[Diagnostic] = []
    if not bundle.is_dir():
        _add(diagnostics, ".", "OKF000", "error", f"bundle directory does not exist: {bundle}")
        return ValidationResult(tuple(diagnostics), 0, 0)

    markdown_files = sorted(bundle.rglob("*.md"))
    concepts = [path for path in markdown_files if path.name not in RESERVED_FILES]
    indexes = [path for path in markdown_files if path.name == "index.md"]
    link_graph: dict[Path, set[Path]] = {}

    if not (bundle / "index.md").exists():
        _add(diagnostics, "index.md", "OKF009", "error", "bundle root index.md is required locally")

    for path in markdown_files:
        _validate_path_shape(path, bundle, diagnostics)
        parsed = parse_markdown(path.read_text(encoding="utf-8-sig"))
        if path.name == "index.md":
            link_graph[path.resolve()] = _validate_index(path, parsed, bundle, diagnostics)
        elif path.name == "log.md":
            _validate_log(path, parsed, bundle, diagnostics)
            link_graph[path.resolve()] = set()
        else:
            link_graph[path.resolve()] = _validate_concept(path, parsed, bundle, repo, diagnostics)

    reachable: set[Path] = set()
    pending = [(bundle / "index.md").resolve()]
    while pending:
        current = pending.pop()
        if current in reachable:
            continue
        reachable.add(current)
        pending.extend(link_graph.get(current, set()) - reachable)

    for concept in concepts:
        if concept.resolve() not in reachable:
            _add(
                diagnostics,
                _display_path(concept, bundle),
                "OKF042",
                "warning",
                "concept is not referenced by an index, concept, or relationship field",
            )

    return ValidationResult(tuple(sorted(diagnostics)), len(concepts), len(indexes))
