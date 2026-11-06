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


def test_no_private_probpipe_name_is_imported():
    for path in source_files():
        for node in ast.walk(ast.parse(path.read_text())):
            for imported in _probpipe_imports(node):
                assert not any(part.startswith("_") for part in imported.split(".")), (
                    f"{path.name}:{node.lineno} imports the private name {imported}"
                )


def test_every_probpipe_function_states_whether_it_is_traceable():
    for path in [*source_files(), *sorted((ROOT / "tests").glob("*.py"))]:
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Call) and _called_name(node) in {"Function", "function"}:
                keywords = {keyword.arg for keyword in node.keywords}
                assert "traceable" in keywords, (
                    f"{path.name}:{node.lineno} builds a Function without stating whether JAX "
                    f"can trace it"
                )


def _probpipe_imports(node):
    """The dotted names a statement imports from ProbPipe, such as ``probpipe.core._x``."""
    if isinstance(node, ast.Import):
        names = [alias.name for alias in node.names]
    elif isinstance(node, ast.ImportFrom) and node.module:
        names = [f"{node.module}.{alias.name}" for alias in node.names]
    else:
        names = []
    return [name for name in names if name.split(".")[0] == "probpipe"]


def _called_name(call):
    function = call.func
    return function.id if isinstance(function, ast.Name) else getattr(function, "attr", None)
