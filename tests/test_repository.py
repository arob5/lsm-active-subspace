"""Rules about the code base itself, checked so they stay true."""

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src" / "lsmas"
REGISTRY = ROOT / "design" / "probpipe-gaps.md"
STOPGAP_TAG = re.compile(r"STOPGAP\(([^)]*)\)")
GAP_ID = re.compile(r"(PG|EK)-\d{2}")


def source_files():
    return sorted(SOURCE.rglob("*.py"))


def registered_gap_ids():
    return set(re.findall(r"^## ((?:PG|EK)-\d{2})$", REGISTRY.read_text(), flags=re.MULTILINE))


def test_every_stopgap_tag_names_a_registered_gap():
    registered = registered_gap_ids()
    for path in source_files():
        for tag in STOPGAP_TAG.findall(path.read_text()):
            ids = [part.strip() for part in tag.split(",")]
            assert ids, f"{path.name}: empty STOPGAP tag"
            for gap_id in ids:
                assert GAP_ID.fullmatch(gap_id), f"{path.name}: malformed gap id {gap_id!r}"
                assert gap_id in registered, f"{path.name}: {gap_id} is not in {REGISTRY.name}"


def test_every_stopgap_module_names_its_gap():
    for path in (SOURCE / "_stopgaps").glob("*.py"):
        if path.name != "__init__.py":
            assert STOPGAP_TAG.search(path.read_text()), f"{path.name} has no STOPGAP tag"


def test_private_probpipe_names_are_imported_in_one_place():
    allowed = SOURCE / "_stopgaps" / "probpipe_private.py"
    for path in source_files():
        if path != allowed:
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.ImportFrom) and node.module and "._" in node.module:
                    assert not node.module.startswith("probpipe"), (
                        f"{path.name} imports the private module {node.module}"
                    )


def test_every_probpipe_function_sets_its_dispatch():
    for path in source_files():
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Call) and _called_name(node) in {"Function", "function"}:
                keywords = {keyword.arg for keyword in node.keywords}
                assert "dispatch" in keywords, (
                    f"{path.name}:{node.lineno} builds a Function without choosing its dispatch"
                )


def _called_name(call):
    function = call.func
    return function.id if isinstance(function, ast.Name) else getattr(function, "attr", None)
