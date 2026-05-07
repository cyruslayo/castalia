"""
Schema-Aware Virtual Filesystem (Notebook 16 + OpenAI Data Agent Insights)

Inspired by OpenAI's multi-layered context system for their data agent:
  Layer 1: Table Usage     -> usage_history, access_count
  Layer 2: Human Annotations -> annotations dict per file
  Layer 3: Code/Metadata   -> auto-inferred schema, derivation_chain
  Layer 5: Memory          -> learned corrections (connected to LTM)

Key design decision: the original VirtualFS was a flat dict {path: content}.
This version adds a metadata registry that tracks every operation, auto-infers
schemas, and maintains data lineage (where did this file come from?).

All operations return structured dicts (never exceptions) for agent consumption.
"""

import csv
import io
import json
import math
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ==============================================================================
# SECTION 1 -- FileMetadata: rich metadata for every file in the filesystem
# ==============================================================================

@dataclass
class ColumnSchema:
    """Schema for a single column in tabular data."""
    name: str
    python_type: str           # "int", "float", "bool", "str", "date", "mixed"
    nullable: bool = False
    null_count: int = 0
    unique_count: int = 0
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    mean_value: Optional[float] = None
    sample_values: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "name": self.name,
            "type": self.python_type,
            "nullable": self.nullable,
            "null_count": self.null_count,
            "unique_count": self.unique_count,
            "sample_values": self.sample_values[:3],  # max 3 samples
        }
        for k in ("min_value", "max_value", "mean_value"):
            v = getattr(self, k)
            if v is not None:
                d[k] = round(v, 4) if isinstance(v, float) else v
        return d


@dataclass
class FileMetadata:
    """
    Rich metadata for a single file.

    Analogous to OpenAI's multi-layer context:
      - schema       -> Layer 3: Code/Metadata (auto-inferred column types)
      - annotations  -> Layer 2: Human Annotations (descriptions, caveats)
      - usage_history-> Layer 1: Table Usage (who accessed, which columns)
      - derivation   -> Layer 3: Pipeline logic (how was this file created?)
    """
    path: str
    content_type: str = "text"          # csv, json, text, binary, unknown
    size_bytes: int = 0
    created_at: float = field(default_factory=time.time)
    modified_at: float = field(default_factory=time.time)
    access_count: int = 0

    # Layer 1: usage tracking
    usage_history: List[Dict[str, Any]] = field(default_factory=list)

    # Layer 2: human / agent annotations
    annotations: Dict[str, Any] = field(default_factory=dict)

    # Layer 3: auto-inferred schema
    schema: Optional[Dict[str, Any]] = None      # top-level keys for JSON, columns for CSV

    # Layer 3: data lineage
    derivation_chain: List[Dict[str, Any]] = field(default_factory=list)

    def record_usage(self, operation: str, details: Optional[Dict] = None) -> None:
        """Log an operation against this file."""
        self.access_count += 1
        entry = {
            "timestamp": time.time(),
            "operation": operation,
            "access_count": self.access_count,
        }
        if details:
            entry.update(details)
        self.usage_history.append(entry)

    def annotate(self, key: str, value: Any) -> None:
        """Add a human or agent annotation."""
        self.annotations[key] = value
        self.modified_at = time.time()

    def record_derivation(self, operation: str, source_path: Optional[str] = None,
                          parameters: Optional[Dict] = None) -> None:
        """Record how this file was created (data lineage)."""
        self.derivation_chain.append({
            "timestamp": time.time(),
            "operation": operation,
            "source": source_path,
            "parameters": parameters or {},
        })

    def to_dict(self) -> Dict[str, Any]:
        """Serialize metadata for LLM consumption."""
        return {
            "path": self.path,
            "content_type": self.content_type,
            "size_bytes": self.size_bytes,
            "access_count": self.access_count,
            "annotations": self.annotations,
            "schema": self.schema,
            "derivation_summary": [d["operation"] for d in self.derivation_chain],
        }


# ==============================================================================
# SECTION 2 -- Schema inference: auto-detect types and stats from raw content
# ==============================================================================

def _try_parse(value: str) -> Tuple[str, Any]:
    """
    Try to parse a string value into a typed value.
    Returns (python_type_name, typed_value).

    Order: bool -> int -> float -> str (fallback)
    """
    v = value.strip()
    if not v:
        return "null", None

    # Boolean detection (OpenAI agent does this for filter correctness)
    vl = v.lower()
    if vl in ("true", "false", "yes", "no", "1", "0"):
        # Disambiguate: "1" could be int, but treat "true"/"false" as bool
        if vl in ("true", "false", "yes", "no"):
            return "bool", vl in ("true", "yes")

    # Integer
    try:
        iv = int(v)
        if str(iv) == v:          # no scientific notation fallback
            return "int", iv
    except ValueError:
        pass

    # Float
    try:
        fv = float(v)
        return "float", fv
    except ValueError:
        pass

    return "str", v


def _infer_csv_schema(rows: List[Dict[str, str]], max_sample: int = 100) -> Dict[str, Any]:
    """
    Infer schema from CSV rows (list of dicts from csv.DictReader).

    Returns: {
      "type": "tabular",
      "columns": {col_name: ColumnSchema.to_dict(), ...},
      "row_count": int,
      "column_count": int,
    }
    """
    if not rows:
        return {"type": "tabular", "columns": {}, "row_count": 0, "column_count": 0}

    columns = list(rows[0].keys())
    col_schemas: Dict[str, ColumnSchema] = {
        c: ColumnSchema(name=c, python_type="unknown") for c in columns
    }

    sample = rows[:max_sample]

    for col in columns:
        types_seen = Counter()
        values = []
        nulls = 0
        for row in sample:
            cell = row.get(col, "")
            ptype, typed = _try_parse(cell)
            if ptype == "null":
                nulls += 1
            else:
                types_seen[ptype] += 1
                values.append(typed)

        # Pick the most common non-null type
        if types_seen:
            dominant = types_seen.most_common(1)[0][0]
        else:
            dominant = "str"

        cs = ColumnSchema(
            name=col,
            python_type=dominant,
            nullable=(nulls > 0),
            null_count=nulls,
            unique_count=len(set(str(v) for v in values)),
            sample_values=[str(v) for v in values[:3]],
        )

        # Numeric stats
        if dominant in ("int", "float"):
            nums = [float(v) for v in values if isinstance(v, (int, float))]
            if nums:
                cs.min_value = min(nums)
                cs.max_value = max(nums)
                cs.mean_value = sum(nums) / len(nums)

        col_schemas[col] = cs

    return {
        "type": "tabular",
        "columns": {c: s.to_dict() for c, s in col_schemas.items()},
        "row_count": len(rows),
        "column_count": len(columns),
    }


def _infer_json_schema(data: Any) -> Dict[str, Any]:
    """
    Infer schema from parsed JSON.

    Returns: {
      "type": "object" | "array" | "scalar",
      "keys": {key: schema} for objects,
      "element_type": schema for arrays,
      "scalar_type": str for scalars,
      "size": int,
    }
    """
    if isinstance(data, dict):
        return {
            "type": "object",
            "keys": {k: _infer_json_schema(v) for k, v in list(data.items())[:20]},
            "size": len(data),
        }
    if isinstance(data, list):
        if not data:
            return {"type": "array", "element_type": None, "size": 0}
        # Infer from first 3 elements, merge schemas heuristically
        sample = data[:3]
        element_schemas = [_infer_json_schema(x) for x in sample]
        # Simplify: just report first element's schema
        return {
            "type": "array",
            "element_type": element_schemas[0] if element_schemas else None,
            "size": len(data),
        }
    if isinstance(data, str):
        return {"type": "scalar", "scalar_type": "str", "size": 1}
    if isinstance(data, bool):
        return {"type": "scalar", "scalar_type": "bool", "size": 1}
    if isinstance(data, int):
        return {"type": "scalar", "scalar_type": "int", "size": 1}
    if isinstance(data, float):
        return {"type": "scalar", "scalar_type": "float", "size": 1}
    return {"type": "scalar", "scalar_type": "unknown", "size": 1}


def _detect_content_type(content: str) -> str:
    """Heuristic content type detection from string content."""
    stripped = content.strip()
    if not stripped:
        return "empty"
    if stripped.startswith("[") or stripped.startswith("{"):
        try:
            json.loads(stripped)
            return "json"
        except json.JSONDecodeError:
            pass
    # CSV heuristic: first line has commas, second line too
    lines = stripped.splitlines()
    if len(lines) >= 2 and all("," in ln for ln in lines[:2]):
        return "csv"
    return "text"


# ==============================================================================
# SECTION 3 -- SchemaAwareFS: production-grade in-memory filesystem
# ==============================================================================

class SchemaAwareFS:
    """
    Dict-based in-memory filesystem with rich metadata tracking.

    Design decisions (informed by OpenAI data agent):
      1. Every file carries metadata (schema, usage, annotations, lineage)
      2. Content type is auto-detected (csv/json/text)
      3. Schema is auto-inferred on create/write, never on read (costly)
      4. All operations return structured dicts (agent-safe)
      5. Operations are logged for debugging and self-correction
    """

    def __init__(self):
        self.files: Dict[str, str] = {}          # path -> content
        self.metadata: Dict[str, FileMetadata] = {}  # path -> FileMetadata

    # --- Core operations ---

    def create(self, path: str, content: str = "") -> Dict[str, Any]:
        """Create a new file. Fails if already exists."""
        if path in self.files:
            return {"success": False, "error": f"File already exists: {path}"}

        self.files[path] = content
        meta = FileMetadata(path=path, size_bytes=len(content))
        self.metadata[path] = meta

        # Auto-infer schema
        ctype = _detect_content_type(content)
        meta.content_type = ctype
        meta.record_derivation("create")

        if ctype == "csv":
            try:
                rows = list(csv.DictReader(io.StringIO(content)))
                meta.schema = _infer_csv_schema(rows)
            except Exception as e:
                meta.annotate("schema_error", str(e))
        elif ctype == "json":
            try:
                data = json.loads(content)
                meta.schema = _infer_json_schema(data)
            except Exception as e:
                meta.annotate("schema_error", str(e))

        meta.record_usage("create")
        return {
            "success": True,
            "status": "created",
            "path": path,
            "size_bytes": len(content),
            "content_type": ctype,
            "schema_inferred": meta.schema is not None,
        }

    def read(self, path: str) -> Dict[str, Any]:
        """Read file contents and return with metadata summary."""
        if path not in self.files:
            return {"success": False, "error": f"File not found: {path}"}

        meta = self.metadata[path]
        meta.record_usage("read")

        return {
            "success": True,
            "path": path,
            "content": self.files[path],
            "size_bytes": len(self.files[path]),
            "content_type": meta.content_type,
            "metadata": meta.to_dict(),
        }

    def write(self, path: str, content: str) -> Dict[str, Any]:
        """Write content (creates or overwrites). Schema re-inferred."""
        is_overwrite = path in self.files
        self.files[path] = content

        if is_overwrite:
            meta = self.metadata[path]
            meta.size_bytes = len(content)
            meta.modified_at = time.time()
            meta.record_derivation("write", parameters={"overwrite": True})
        else:
            meta = FileMetadata(path=path, size_bytes=len(content))
            self.metadata[path] = meta
            meta.record_derivation("create")

        ctype = _detect_content_type(content)
        meta.content_type = ctype
        meta.schema = None  # clear old schema

        if ctype == "csv":
            try:
                rows = list(csv.DictReader(io.StringIO(content)))
                meta.schema = _infer_csv_schema(rows)
            except Exception as e:
                meta.annotate("schema_error", str(e))
        elif ctype == "json":
            try:
                data = json.loads(content)
                meta.schema = _infer_json_schema(data)
            except Exception as e:
                meta.annotate("schema_error", str(e))

        meta.record_usage("write")
        return {
            "success": True,
            "status": "written" if is_overwrite else "created",
            "path": path,
            "size_bytes": len(content),
            "content_type": ctype,
            "schema_inferred": meta.schema is not None,
        }

    def append(self, path: str, content: str) -> Dict[str, Any]:
        """Append to existing file. Schema is NOT re-inferred (costly)."""
        if path not in self.files:
            return {"success": False, "error": f"File not found: {path}"}

        self.files[path] += content
        meta = self.metadata[path]
        meta.size_bytes = len(self.files[path])
        meta.modified_at = time.time()
        meta.record_derivation("append")
        meta.record_usage("append")

        # Mark schema as stale
        if meta.schema:
            meta.annotate("schema_stale", True)

        return {
            "success": True,
            "status": "appended",
            "path": path,
            "size_bytes": meta.size_bytes,
        }

    def delete(self, path: str) -> Dict[str, Any]:
        """Delete a file."""
        if path not in self.files:
            return {"success": False, "error": f"File not found: {path}"}

        del self.files[path]
        del self.metadata[path]
        return {"success": True, "status": "deleted", "path": path}

    def list_files(self, prefix: str = "") -> Dict[str, Any]:
        """List files, optionally filtered by prefix."""
        matches = sorted(p for p in self.files if p.startswith(prefix))
        return {
            "success": True,
            "files": matches,
            "count": len(matches),
            "prefix": prefix or "/",
        }

    def exists(self, path: str) -> Dict[str, Any]:
        """Check existence."""
        return {"success": True, "exists": path in self.files, "path": path}

    def stats(self) -> Dict[str, Any]:
        """Global filesystem statistics."""
        total_bytes = sum(len(c) for c in self.files.values())
        type_counts = Counter(m.content_type for m in self.metadata.values())
        return {
            "success": True,
            "total_files": len(self.files),
            "total_bytes": total_bytes,
            "content_type_breakdown": dict(type_counts),
            "most_accessed": sorted(
                self.metadata.values(),
                key=lambda m: m.access_count,
                reverse=True,
            )[:5],
        }

    # --- Advanced: derivation and lineage ---

    def derive(self, source_path: str, dest_path: str,
               operation: str, result_content: str,
               parameters: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Create a derived file, recording full lineage.

        Example: derive("/data/countries.csv", "/data/asia_only.csv",
                        "csv_filter", content, {"continent": "Asia"})
        This lets the agent answer "where did this file come from?"
        """
        # Create destination
        result = self.create(dest_path, result_content)
        if not result.get("success"):
            return result

        # Record lineage on destination
        dest_meta = self.metadata.get(dest_path)
        if dest_meta:
            dest_meta.record_derivation(operation, source_path, parameters)

        # Record that source was used to produce something
        src_meta = self.metadata.get(source_path)
        if src_meta:
            src_meta.annotate("produced", src_meta.annotations.get("produced", []) + [{
                "dest": dest_path,
                "operation": operation,
                "timestamp": time.time(),
            }])

        return result

    def get_lineage(self, path: str) -> Dict[str, Any]:
        """Return full derivation chain for a file."""
        meta = self.metadata.get(path)
        if not meta:
            return {"success": False, "error": f"File not found: {path}"}
        return {
            "success": True,
            "path": path,
            "derivation_chain": meta.derivation_chain,
            "annotations": meta.annotations,
        }

    def get_popular_columns(self, path: str) -> Dict[str, Any]:
        """
        Which columns have been queried most often?
        Informed by OpenAI Layer 1 (Table Usage).
        """
        meta = self.metadata.get(path)
        if not meta:
            return {"success": False, "error": f"File not found: {path}"}

        col_counts = Counter()
        for entry in meta.usage_history:
            cols = entry.get("columns")
            if cols:
                for c in (cols if isinstance(cols, list) else [cols]):
                    col_counts[c] += 1

        return {
            "success": True,
            "path": path,
            "popular_columns": col_counts.most_common(10),
            "total_queries": len(meta.usage_history),
        }


# ==============================================================================
# SECTION 4 -- Fast tests
# ==============================================================================

def run_tests():
    """Validate the SchemaAwareFS."""
    fs = SchemaAwareFS()

    # Test 1: Create CSV and infer schema
    csv_content = """name,age,city,salary
Alice,30,New York,85000
Bob,25,San Francisco,92000
Charlie,35,New York,78000
Diana,28,Chicago,71000"""
    r = fs.create("/data/employees.csv", csv_content)
    assert r["success"], r
    assert r["content_type"] == "csv"
    meta = fs.metadata["/data/employees.csv"]
    assert meta.schema is not None
    assert meta.schema["type"] == "tabular"
    assert "name" in meta.schema["columns"]
    assert meta.schema["columns"]["age"]["type"] == "int"
    assert meta.schema["columns"]["salary"]["type"] == "int"
    print("[PASS] Test 1: CSV schema inference")

    # Test 2: Create JSON and infer schema
    json_content = json.dumps({
        "company": "TechCorp",
        "founded": 2015,
        "departments": [
            {"name": "Engineering", "headcount": 45},
            {"name": "Marketing", "headcount": 12}
        ]
    })
    r = fs.create("/data/company.json", json_content)
    assert r["success"]
    assert r["content_type"] == "json"
    meta = fs.metadata["/data/company.json"]
    assert meta.schema["type"] == "object"
    assert "departments" in meta.schema["keys"]
    print("[PASS] Test 2: JSON schema inference")

    # Test 3: Read returns metadata
    r = fs.read("/data/employees.csv")
    assert r["success"]
    assert "metadata" in r
    assert r["metadata"]["access_count"] == 2  # create + read
    print("[PASS] Test 3: Read with metadata")

    # Test 4: Derive with lineage
    r = fs.derive("/data/employees.csv", "/data/ny_only.csv",
                    "csv_filter", "name,age,city,salary\nAlice,30,New York,85000\n",
                    {"city": "New York"})
    assert r["success"]
    lineage = fs.get_lineage("/data/ny_only.csv")
    assert lineage["success"]
    assert len(lineage["derivation_chain"]) >= 1
    assert lineage["derivation_chain"][-1]["source"] == "/data/employees.csv"
    print("[PASS] Test 4: Data lineage tracking")

    # Test 5: Popular columns
    # Simulate usage with column tracking
    fs.metadata["/data/employees.csv"].record_usage("query", {"columns": ["salary"]})
    fs.metadata["/data/employees.csv"].record_usage("query", {"columns": ["salary", "age"]})
    pop = fs.get_popular_columns("/data/employees.csv")
    assert pop["success"]
    assert pop["popular_columns"][0][0] == "salary"
    print("[PASS] Test 5: Popular column tracking")

    # Test 6: Error handling
    r = fs.read("/nonexistent")
    assert not r["success"]
    assert "error" in r
    print("[PASS] Test 6: Error handling")

    # Test 7: Stats
    r = fs.stats()
    assert r["success"]
    assert r["total_files"] == 3
    print("[PASS] Test 7: Global stats")

    # Test 8: Schema stale after append
    fs.append("/data/employees.csv", "\nEve,29,Chicago,73000")
    meta = fs.metadata["/data/employees.csv"]
    assert meta.annotations.get("schema_stale") is True
    print("[PASS] Test 8: Schema staleness after append")

    print("\n*** All 8 SchemaAwareFS tests passed!")


if __name__ == "__main__":
    run_tests()
