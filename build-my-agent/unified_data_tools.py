"""
Unified Data Tools (Notebook 16 + OpenAI Data Agent Insights)

Original Notebook 16 had 6 separate tool classes:
  CSVTools:     csv_read, csv_query, csv_filter, csv_statistics, csv_sort, csv_group_by
  JSONTools:    json_read, json_query, json_transform, json_flatten
  StatisticsTools: mean, median, mode, std_dev, percentile, correlation, regression, summary

OpenAI Lesson #1: "Less is More". Overlapping tools confuse agents.
We consolidate into 4 unified interfaces:
  DataReader   -- parse CSV/JSON from filesystem or string, auto-infer schema
  DataQuery    -- unified filter/sort/group/aggregate (SQL-like, multi-condition)
  DataStats    -- comprehensive statistics with auto-type detection
  DataTransform -- derive, select, rename, reshape

All methods accept either a filesystem result OR raw data for composability.
All methods return structured dicts with {"success", "error", ...}.
"""

import csv
import io
import json
import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union


# ==============================================================================
# SECTION 1 -- DataReader: parse CSV/JSON from filesystem or raw string
# ==============================================================================

class DataReader:
    """
    Read and parse data files. Returns structured results with schema info.

    Supports two input modes:
      1. fs_result: pass the result dict from SchemaAwareFS.read()
      2. raw_content: pass a string directly (useful for inline data)
    """

    @staticmethod
    def read_csv(content: str) -> Dict[str, Any]:
        """Parse CSV string into typed rows with column info."""
        try:
            reader = csv.DictReader(io.StringIO(content))
            columns = reader.fieldnames or []
            rows_raw = list(reader)
            # Type-infer each row (same logic as SchemaAwareFS)
            rows = []
            for r in rows_raw:
                typed_row = {}
                for col, val in r.items():
                    ptype, typed = _try_parse(val.strip())
                    if ptype == "null":
                        typed_row[col] = None
                    elif ptype == "bool":
                        typed_row[col] = typed
                    elif ptype == "int":
                        typed_row[col] = typed
                    elif ptype == "float":
                        typed_row[col] = typed
                    else:
                        typed_row[col] = val
                rows.append(typed_row)

            return {
                "success": True,
                "type": "tabular",
                "columns": columns,
                "row_count": len(rows),
                "rows": rows,
                "preview": rows[:3],
            }
        except Exception as e:
            return {"success": False, "error": f"CSV parse error: {e}"}

    @staticmethod
    def read_json(content: str) -> Dict[str, Any]:
        """Parse JSON string into structured data."""
        try:
            data = json.loads(content)
            dtype = type(data).__name__
            size = len(data) if isinstance(data, (list, dict)) else 1
            return {
                "success": True,
                "type": "json",
                "data": data,
                "json_type": dtype,
                "size": size,
            }
        except json.JSONDecodeError as e:
            return {"success": False, "error": f"JSON parse error: {e}"}

    @classmethod
    def read(cls, fs_result_or_content: Union[Dict[str, Any], str]) -> Dict[str, Any]:
        """
        Universal read dispatcher.
        If dict with 'content' key (from SchemaAwareFS), auto-detect type.
        If string, auto-detect type.
        """
        if isinstance(fs_result_or_content, dict):
            if not fs_result_or_content.get("success"):
                return fs_result_or_content
            content = fs_result_or_content.get("content", "")
            ctype = fs_result_or_content.get("content_type", "text")
            if ctype == "csv":
                result = cls.read_csv(content)
                result["source"] = fs_result_or_content.get("path", "unknown")
                return result
            if ctype == "json":
                result = cls.read_json(content)
                result["source"] = fs_result_or_content.get("path", "unknown")
                return result
            return {"success": False, "error": f"Unsupported content_type: {ctype}"}

        # Raw string: auto-detect
        s = fs_result_or_content.strip()
        if s.startswith("[") or s.startswith("{"):
            # Try JSON first
            r = cls.read_json(fs_result_or_content)
            if r["success"]:
                return r
            # Fall back to CSV
            if "\n" in s and "," in s.split("\n")[0]:
                return cls.read_csv(fs_result_or_content)
        if "\n" in s and "," in s.split("\n")[0]:
            return cls.read_csv(fs_result_or_content)
        return {"success": False, "error": "Could not auto-detect format (not CSV or JSON)"}


# ==============================================================================
# SECTION 2 -- DataQuery: unified filter/sort/group/aggregate/projection
# ==============================================================================

class DataQuery:
    """
    Unified query interface for tabular data.

    A single method handles: filter, sort, group-by, aggregate, projection, limit.
    This avoids the agent having to call 3 separate tools (filter -> sort -> group).

    Conditions format: [{"column": "col", "operator": "==|!=|>|<|>=|<=|contains", "value": ...}, ...]
    Sort format: [{"column": "col", "descending": True/False}, ...] or "col" or ["col1", "-col2"]
    Group format: {"by": "col", "aggregates": [{"column": "col2", "func": "sum|count|mean|min|max"}]}
    Projection: ["col1", "col2"] (if None, all columns)
    Limit: int
    """

    @staticmethod
    def query(data: Dict[str, Any],
              conditions: Optional[List[Dict[str, Any]]] = None,
              sort_by: Optional[Union[str, List[str], List[Dict[str, Any]]]] = None,
              group_by: Optional[Dict[str, Any]] = None,
              select: Optional[List[str]] = None,
              limit: Optional[int] = None,
              ) -> Dict[str, Any]:
        """
        Execute a unified query on tabular data.

        Returns: {
          "success": True/False,
          "rows": [...],           # result rows (already projected)
          "row_count": int,
          "columns": [...],        # column names in output
          "query_summary": str,    # human-readable description
        }
        """
        if not data.get("success"):
            return data
        if data.get("type") != "tabular":
            return {"success": False, "error": "DataQuery only supports tabular data"}

        rows = data.get("rows", [])
        columns = list(data.get("columns", []))

        # --- Step 1: Normalize + Filter ---
        if conditions:
            conditions = _normalize_conditions(conditions)
            rows = _apply_conditions(rows, conditions)

        # --- Step 2: Group (if requested) ---
        if group_by:
            group_result = _apply_group_by(rows, group_by)
            # Group result is a new table with grouped columns
            rows = group_result["rows"]
            columns = list(group_result["columns"])

        # --- Step 3: Sort ---
        if sort_by:
            rows = _apply_sort(rows, sort_by)

        # --- Step 4: Projection ---
        if select:
            rows = [{k: r.get(k) for k in select if k in r} for r in rows]
            columns = [c for c in select if c in columns]

        # --- Step 5: Limit ---
        if limit:
            rows = rows[:limit]

        # Build summary
        parts = []
        if conditions:
            parts.append(f"filtered by {len(conditions)} condition(s)")
        if group_by:
            parts.append(f"grouped by {group_by.get('by')}")
        if sort_by:
            parts.append("sorted")
        if select:
            parts.append(f"projected to {len(select)} column(s)")
        if limit:
            parts.append(f"limited to {limit}")
        summary = "; ".join(parts) if parts else "full dataset"

        return {
            "success": True,
            "type": "tabular",
            "rows": rows,
            "row_count": len(rows),
            "columns": columns,
            "query_summary": summary,
            "source": data.get("source", "unknown"),
        }


# ==============================================================================
# SECTION 3 -- DataStats: comprehensive statistics with auto-type detection
# ==============================================================================

class DataStats:
    """
    Compute statistics on tabular data columns.

    Auto-detects numeric columns. Supports single-column summary, two-column
correlation, and linear regression.
    """

    @staticmethod
    def summarize(data: Dict[str, Any],
                  columns: Optional[Union[str, List[str]]] = None,
                  percentiles: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        Comprehensive summary of one or more columns.

        Returns: {
          "success": True,
          "summaries": {
            "col_name": {"count", "mean", "median", "std_dev", "min", "max",
                         "p25", "p75", "iqr", "null_count", "type"},
            ...
          }
        }
        """
        if not data.get("success"):
            return data
        if data.get("type") != "tabular":
            return {"success": False, "error": "DataStats only supports tabular data"}

        rows = data.get("rows", [])
        all_cols = data.get("columns", [])

        target_cols = [columns] if isinstance(columns, str) else columns
        if not target_cols:
            # Auto-detect numeric columns
            target_cols = []
            for c in all_cols:
                if any(isinstance(r.get(c), (int, float)) for r in rows[:20]):
                    target_cols.append(c)

        summaries = {}
        for col in target_cols:
            if col not in all_cols:
                summaries[col] = {"error": f"Column '{col}' not found"}
                continue

            vals = [r.get(col) for r in rows if r.get(col) is not None]
            numeric = [float(v) for v in vals if isinstance(v, (int, float))]

            if not numeric:
                # String mode: frequency counts
                freq = Counter(str(v) for v in vals)
                summaries[col] = {
                    "type": "categorical",
                    "count": len(vals),
                    "null_count": len(rows) - len(vals),
                    "unique_count": len(freq),
                    "most_common": freq.most_common(3),
                }
                continue

            n = len(numeric)
            s = sorted(numeric)
            mean = sum(numeric) / n
            variance = sum((x - mean) ** 2 for x in numeric) / n
            std_dev = math.sqrt(variance)
            median = _median(numeric)

            pcts = percentiles or [25, 75]
            pct_values = {f"p{p}": _percentile(numeric, p) for p in pcts}

            summaries[col] = {
                "type": "numeric",
                "count": n,
                "null_count": len(rows) - len(vals),
                "mean": round(mean, 4),
                "median": round(median, 4),
                "std_dev": round(std_dev, 4),
                "min": round(min(numeric), 4),
                "max": round(max(numeric), 4),
                "range": round(max(numeric) - min(numeric), 4),
                **{k: round(v, 4) for k, v in pct_values.items()},
            }
            if "p25" in summaries[col] and "p75" in summaries[col]:
                summaries[col]["iqr"] = round(
                    summaries[col]["p75"] - summaries[col]["p25"], 4
                )

        return {
            "success": True,
            "type": "statistics",
            "summaries": summaries,
            "source": data.get("source", "unknown"),
        }

    @staticmethod
    def correlate(data: Dict[str, Any],
                  x_column: str, y_column: str) -> Dict[str, Any]:
        """Pearson correlation coefficient between two numeric columns."""
        if not data.get("success"):
            return data
        rows = data.get("rows", [])
        x_vals = [float(r[x_column]) for r in rows
                  if r.get(x_column) is not None and r.get(y_column) is not None
                  and isinstance(r[x_column], (int, float))]
        y_vals = [float(r[y_column]) for r in rows
                  if r.get(x_column) is not None and r.get(y_column) is not None
                  and isinstance(r[y_column], (int, float))]

        if len(x_vals) != len(y_vals) or len(x_vals) < 2:
            return {"success": False, "error": "Need at least 2 matching numeric pairs"}

        n = len(x_vals)
        x_mean = sum(x_vals) / n
        y_mean = sum(y_vals) / n
        num = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_vals, y_vals))
        den_x = math.sqrt(sum((x - x_mean) ** 2 for x in x_vals))
        den_y = math.sqrt(sum((y - y_mean) ** 2 for y in y_vals))

        if den_x == 0 or den_y == 0:
            return {"success": False, "error": "No variance in one or both variables"}

        r = num / (den_x * den_y)
        return {
            "success": True,
            "type": "correlation",
            "x_column": x_column,
            "y_column": y_column,
            "correlation": round(r, 4),
            "n": n,
            "interpretation": _interpret_correlation(r),
        }

    @staticmethod
    def regress(data: Dict[str, Any],
                x_column: str, y_column: str) -> Dict[str, Any]:
        """Simple linear regression: y = slope*x + intercept."""
        if not data.get("success"):
            return data
        rows = data.get("rows", [])
        x_vals = [float(r[x_column]) for r in rows
                  if r.get(x_column) is not None and r.get(y_column) is not None
                  and isinstance(r[x_column], (int, float))]
        y_vals = [float(r[y_column]) for r in rows
                  if r.get(x_column) is not None and r.get(y_column) is not None
                  and isinstance(r[y_column], (int, float))]

        if len(x_vals) != len(y_vals) or len(x_vals) < 2:
            return {"success": False, "error": "Need at least 2 matching numeric pairs"}

        n = len(x_vals)
        x_mean = sum(x_vals) / n
        y_mean = sum(y_vals) / n
        num = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_vals, y_vals))
        den = sum((x - x_mean) ** 2 for x in x_vals)

        if den == 0:
            return {"success": False, "error": "No variance in x values"}

        slope = num / den
        intercept = y_mean - slope * x_mean

        # R-squared
        y_pred = [slope * x + intercept for x in x_vals]
        ss_res = sum((y - yp) ** 2 for y, yp in zip(y_vals, y_pred))
        ss_tot = sum((y - y_mean) ** 2 for y in y_vals)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

        return {
            "success": True,
            "type": "regression",
            "equation": f"y = {round(slope, 4)}x + {round(intercept, 4)}",
            "slope": round(slope, 4),
            "intercept": round(intercept, 4),
            "r_squared": round(r_squared, 4),
            "n": n,
        }


# ==============================================================================
# SECTION 4 -- DataTransform: derive, select, rename, reshape
# ==============================================================================

class DataTransform:
    """Transform tabular data: derive columns, select, rename, pivot."""

    @staticmethod
    def derive(data: Dict[str, Any], new_column: str,
               expression: Union[str, Callable]) -> Dict[str, Any]:
        """
        Add a derived column from an expression.

        expression can be:
          - a string with column placeholders: "{gdp} * 1000 / {population}"
          - a callable: lambda row: row['gdp'] / row['population']
        """
        if not data.get("success"):
            return data
        if data.get("type") != "tabular":
            return {"success": False, "error": "DataTransform only supports tabular data"}

        rows = data.get("rows", [])
        columns = list(data.get("columns", []))
        new_rows = []

        if callable(expression):
            for r in rows:
                nr = dict(r)
                try:
                    nr[new_column] = expression(r)
                except Exception as e:
                    nr[new_column] = None
                new_rows.append(nr)
        else:
            # String expression with placeholders
            for r in rows:
                nr = dict(r)
                try:
                    formatted = expression.format(**{k: (v if v is not None else 0) for k, v in r.items()})
                    nr[new_column] = _safe_eval(formatted)
                except Exception:
                    nr[new_column] = None
                new_rows.append(nr)

        if new_column not in columns:
            columns.append(new_column)

        return {
            "success": True,
            "type": "tabular",
            "rows": new_rows,
            "row_count": len(new_rows),
            "columns": columns,
            "source": data.get("source", "unknown"),
            "derivation": f"derived '{new_column}' from expression",
        }

    @staticmethod
    def select(data: Dict[str, Any], columns: List[str]) -> Dict[str, Any]:
        """Select specific columns (projection)."""
        if not data.get("success"):
            return data
        if data.get("type") != "tabular":
            return {"success": False, "error": "DataTransform only supports tabular data"}

        rows = data.get("rows", [])
        new_rows = [{k: r.get(k) for k in columns} for r in rows]
        return {
            "success": True,
            "type": "tabular",
            "rows": new_rows,
            "row_count": len(new_rows),
            "columns": columns,
            "source": data.get("source", "unknown"),
        }

    @staticmethod
    def rename(data: Dict[str, Any], mapping: Dict[str, str]) -> Dict[str, Any]:
        """Rename columns using {old_name: new_name} mapping."""
        if not data.get("success"):
            return data
        if data.get("type") != "tabular":
            return {"success": False, "error": "DataTransform only supports tabular data"}

        rows = data.get("rows", [])
        old_cols = data.get("columns", [])
        new_cols = [mapping.get(c, c) for c in old_cols]
        new_rows = []
        for r in rows:
            nr = {mapping.get(k, k): v for k, v in r.items()}
            new_rows.append(nr)

        return {
            "success": True,
            "type": "tabular",
            "rows": new_rows,
            "row_count": len(new_rows),
            "columns": new_cols,
            "source": data.get("source", "unknown"),
        }

    @staticmethod
    def pivot(data: Dict[str, Any],
              index: str, columns: str, values: str,
              agg_func: str = "sum") -> Dict[str, Any]:
        """
        Simple pivot: reshape long-to-wide.
        index = row identifier column
        columns = column to pivot into new headers
        values = values to aggregate
        agg_func = sum|mean|count|min|max
        """
        if not data.get("success"):
            return data
        if data.get("type") != "tabular":
            return {"success": False, "error": "DataTransform only supports tabular data"}

        rows = data.get("rows", [])
        # Group by (index, columns)
        grouped = defaultdict(list)
        for r in rows:
            key = (r.get(index), r.get(columns))
            grouped[key].append(r.get(values))

        # Extract unique index and column values
        index_vals = sorted(set(k[0] for k in grouped))
        col_vals = sorted(set(k[1] for k in grouped))

        new_rows = []
        new_columns = [index] + [str(cv) for cv in col_vals]
        for iv in index_vals:
            row = {index: iv}
            for cv in col_vals:
                vals = [v for v in grouped.get((iv, cv), []) if v is not None]
                if not vals:
                    row[str(cv)] = None
                elif agg_func == "sum":
                    row[str(cv)] = sum(float(v) for v in vals if isinstance(v, (int, float)))
                elif agg_func == "mean":
                    row[str(cv)] = sum(float(v) for v in vals) / len(vals)
                elif agg_func == "count":
                    row[str(cv)] = len(vals)
                elif agg_func == "min":
                    row[str(cv)] = min(float(v) for v in vals)
                elif agg_func == "max":
                    row[str(cv)] = max(float(v) for v in vals)
                else:
                    row[str(cv)] = sum(float(v) for v in vals)
            new_rows.append(row)

        return {
            "success": True,
            "type": "tabular",
            "rows": new_rows,
            "row_count": len(new_rows),
            "columns": new_columns,
            "source": data.get("source", "unknown"),
        }


# ==============================================================================
# SECTION 5 -- Internal helpers
# ==============================================================================

def _try_parse(value: str) -> Tuple[str, Any]:
    """Parse a string value. Returns (type_name, typed_value)."""
    v = value.strip()
    if not v:
        return "null", None
    vl = v.lower()
    if vl in ("true", "false", "yes", "no"):
        return "bool", vl in ("true", "yes")
    try:
        iv = int(v)
        if str(iv) == v:
            return "int", iv
    except ValueError:
        pass
    try:
        fv = float(v)
        return "float", fv
    except ValueError:
        pass
    return "str", v


def _safe_eval(expr: str) -> Any:
    """Safely evaluate a simple arithmetic expression."""
    allowed_names = {"abs": abs, "round": round, "max": max, "min": min, "sum": sum}
    try:
        return eval(expr, {"__builtins__": {}}, allowed_names)
    except Exception as e:
        return f"<eval_error: {e}>"


def _apply_conditions(rows: List[Dict], conditions: List[Dict]) -> List[Dict]:
    """Apply multiple filter conditions (AND logic)."""
    result = rows
    for cond in conditions:
        col = cond.get("column")
        op = cond.get("operator", "==")
        val = cond.get("value")
        result = [r for r in result if _matches_condition(r, col, op, val)]
    return result


def _matches_condition(row: Dict, col: str, op: str, val: Any) -> bool:
    """Check if a row matches a single condition."""
    cell = row.get(col)
    if cell is None:
        return False

    # Try numeric comparison first
    try:
        cell_num = float(cell)
        val_num = float(val)
        if op == "==": return abs(cell_num - val_num) < 1e-9
        if op == "!=": return abs(cell_num - val_num) >= 1e-9
        if op == ">": return cell_num > val_num
        if op == "<": return cell_num < val_num
        if op == ">=": return cell_num >= val_num
        if op == "<=": return cell_num <= val_num
    except (ValueError, TypeError):
        pass

    # String comparison
    cell_s = str(cell).lower()
    val_s = str(val).lower()
    if op == "==": return cell_s == val_s
    if op == "!=": return cell_s != val_s
    if op == "contains": return val_s in cell_s
    return False


def _normalize_conditions(conditions: Any) -> Optional[List[Dict[str, Any]]]:
    """Normalize LLM-produced conditions into the expected list-of-dicts format.

    Handles:
      - str: "continent == Asia" -> [{"column": "continent", "operator": "==", "value": "Asia"}]
      - dict: {"column": "x", ...} -> [{...}]
      - list of str: ["age > 30", ...] -> [{...}, ...]
      - list of dict: as-is
      - None: None
    """
    if conditions is None:
        return None

    # Single dict -> wrap in list
    if isinstance(conditions, dict):
        return [conditions]

    # Single string -> parse into list
    if isinstance(conditions, str):
        parsed = _parse_condition_string(conditions)
        return [parsed] if parsed else None

    # List -> normalize each element
    if isinstance(conditions, list):
        result = []
        for item in conditions:
            if isinstance(item, dict):
                result.append(item)
            elif isinstance(item, str):
                parsed = _parse_condition_string(item)
                if parsed:
                    result.append(parsed)
            # Skip unknown types
        return result if result else None

    return None


def _parse_condition_string(s: str) -> Optional[Dict[str, Any]]:
    """Parse a string condition like 'column == value' into a dict."""
    import re
    # Match: column operator value (operators: ==, !=, >, <, >=, <=, contains)
    # Value may be quoted or bare
    m = re.match(
        r'^\s*([A-Za-z_][A-Za-z0-9_]*)\s*'
        r'(==|!=|>=|<=|>|<|contains)\s*'
        r'(?:["\']?(.*?)["\']?)?\s*$',
        s.strip()
    )
    if m:
        return {
            "column": m.group(1),
            "operator": m.group(2),
            "value": m.group(3) if m.group(3) is not None else "",
        }
    return None


def _apply_sort(rows: List[Dict], sort_by: Any) -> List[Dict]:
    """Apply sort specification to rows."""
    # Normalize to list of (column, descending) tuples
    specs = []
    if isinstance(sort_by, str):
        specs = [(sort_by, False)]
    elif isinstance(sort_by, list):
        if sort_by and isinstance(sort_by[0], dict):
            specs = [(s["column"], s.get("descending", False)) for s in sort_by]
        else:
            for s in sort_by:
                if isinstance(s, str) and s.startswith("-"):
                    specs.append((s[1:], True))
                else:
                    specs.append((s, False))

    def sort_key(row):
        keys = []
        for col, desc in reversed(specs):
            val = row.get(col)
            try:
                keys.append((0, float(val) if val is not None else float("-inf")))
            except (ValueError, TypeError):
                keys.append((1, str(val) if val is not None else ""))
            if desc:
                keys[-1] = (keys[-1][0], -keys[-1][1] if isinstance(keys[-1][1], (int, float)) else keys[-1][1])
        return tuple(k[1] for k in keys)

    return sorted(rows, key=sort_key)


def _apply_group_by(rows: List[Dict], group_by: Dict) -> Dict[str, Any]:
    """Apply group-by with aggregates."""
    group_col = group_by.get("by")
    aggregates = group_by.get("aggregates", [])

    groups = defaultdict(list)
    for r in rows:
        key = r.get(group_col, "unknown")
        groups[key].append(r)

    result_rows = []
    result_columns = [group_col]

    for gkey in sorted(groups):
        group_rows = groups[gkey]
        result_row = {group_col: gkey}
        for agg in aggregates:
            col = agg["column"]
            func = agg["func"]
            alias = agg.get("alias", f"{func}_{col}")
            vals = [float(r[col]) for r in group_rows
                    if r.get(col) is not None and isinstance(r[col], (int, float))]
            if not vals:
                result_row[alias] = None
            elif func == "sum":
                result_row[alias] = round(sum(vals), 4)
            elif func == "count":
                result_row[alias] = len(vals)
            elif func == "mean":
                result_row[alias] = round(sum(vals) / len(vals), 4)
            elif func == "min":
                result_row[alias] = round(min(vals), 4)
            elif func == "max":
                result_row[alias] = round(max(vals), 4)
            else:
                result_row[alias] = round(sum(vals), 4)
            if alias not in result_columns:
                result_columns.append(alias)
        result_rows.append(result_row)

    return {"rows": result_rows, "columns": result_columns}


def _median(values: List[float]) -> float:
    s = sorted(values)
    n = len(s)
    if n % 2 == 0:
        return (s[n // 2 - 1] + s[n // 2]) / 2
    return s[n // 2]


def _percentile(values: List[float], p: float) -> float:
    s = sorted(values)
    k = (len(s) - 1) * (p / 100)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return s[int(k)]
    return s[f] * (c - k) + s[c] * (k - f)


def _interpret_correlation(r: float) -> str:
    ar = abs(r)
    if ar < 0.1: return "negligible"
    if ar < 0.3: return "weak"
    if ar < 0.5: return "moderate"
    if ar < 0.7: return "strong"
    return "very strong"


# ==============================================================================
# SECTION 6 -- Fast tests
# ==============================================================================

def run_tests():
    """Run fast deterministic tests on all 4 tool classes."""
    reader = DataReader()

    # Sample CSV
    csv_content = """name,age,city,salary
Alice,30,New York,85000
Bob,25,San Francisco,92000
Charlie,35,New York,78000
Diana,28,Chicago,71000
Eve,32,San Francisco,95000"""

    # Test 1: Read CSV with auto-typing
    data = reader.read(csv_content)
    assert data["success"], data
    assert data["type"] == "tabular"
    assert data["row_count"] == 5
    assert data["rows"][0]["age"] == 30        # auto-typed int
    assert data["rows"][0]["salary"] == 85000  # auto-typed int
    print("[PASS] Test 1: Read CSV with auto-typing")

    # Test 2: Read JSON
    json_content = json.dumps({"users": [{"name": "Alice", "score": 95}, {"name": "Bob", "score": 87}]})
    jdata = reader.read(json_content)
    assert jdata["success"]
    assert jdata["json_type"] == "dict"
    print("[PASS] Test 2: Read JSON")

    # Test 3: Unified query - filter + sort + limit
    q = DataQuery()
    result = q.query(data,
                     conditions=[{"column": "city", "operator": "==", "value": "New York"}],
                     sort_by="-salary",
                     select=["name", "salary"],
                     limit=1)
    assert result["success"]
    assert result["row_count"] == 1
    assert result["rows"][0]["name"] == "Alice"
    print("[PASS] Test 3: Unified query (filter + sort + limit)")

    # Test 4: Group by with aggregates
    result = q.query(data, group_by={
        "by": "city",
        "aggregates": [
            {"column": "salary", "func": "mean", "alias": "avg_salary"},
            {"column": "salary", "func": "count", "alias": "count"},
        ]
    })
    assert result["success"]
    assert result["row_count"] == 3  # New York, San Francisco, Chicago
    cities = {r["city"]: r for r in result["rows"]}
    assert cities["San Francisco"]["avg_salary"] == 93500.0
    print("[PASS] Test 4: Group-by with aggregates")

    # Test 5: Statistics auto-detect numeric
    stats = DataStats()
    s = stats.summarize(data)
    assert s["success"]
    assert "age" in s["summaries"]
    assert s["summaries"]["age"]["type"] == "numeric"
    # Categorical must be requested explicitly (auto-detect skips non-numeric)
    s2 = stats.summarize(data, columns=["city"])
    assert s2["summaries"]["city"]["type"] == "categorical"
    print("[PASS] Test 5: Auto-detect numeric stats + explicit categorical")

    # Test 6: Correlation
    c = stats.correlate(data, "age", "salary")
    assert c["success"]
    assert "correlation" in c
    print("[PASS] Test 6: Correlation")

    # Test 7: Regression
    reg = stats.regress(data, "age", "salary")
    assert reg["success"]
    assert "r_squared" in reg
    print("[PASS] Test 7: Regression")

    # Test 8: Derive column
    xf = DataTransform()
    derived = xf.derive(data, "salary_k", "{salary} / 1000")
    assert derived["success"]
    assert "salary_k" in derived["columns"]
    assert derived["rows"][0]["salary_k"] == 85.0
    print("[PASS] Test 8: Derive column")

    # Test 9: Rename columns
    renamed = xf.rename(derived, {"salary_k": "salary_thousands"})
    assert renamed["success"]
    assert "salary_thousands" in renamed["columns"]
    print("[PASS] Test 9: Rename columns")

    # Test 10: Pivot
    # Need data with repeated index values
    pivot_data = reader.read("""month,product,revenue
Q1,A,100
Q1,B,200
Q2,A,150
Q2,B,250""")
    pivoted = xf.pivot(pivot_data, index="month", columns="product", values="revenue", agg_func="sum")
    assert pivoted["success"]
    assert "A" in pivoted["columns"]
    assert "B" in pivoted["columns"]
    q1 = [r for r in pivoted["rows"] if r["month"] == "Q1"][0]
    assert q1["A"] == 100
    assert q1["B"] == 200
    print("[PASS] Test 10: Pivot")

    print("\n*** All 10 UnifiedDataTools tests passed!")


if __name__ == "__main__":
    run_tests()
