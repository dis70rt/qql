from typing import Tuple, List, Callable
from handler.online.query_sampler import SamplePlan
from parser import ParseResult
import math
import re

# def _rewrite_agg_weighted(agg: AggregateInfo, weight_expr: str):
#     fn = agg.func.lower()
#     col = agg.column or ''
#     alias = getattr(agg, "alias", None) or None
#     if fn == "sum":
#         expr = f"SUM(({col}) * {weight_expr})"
#     elif fn == "count":
#         if not col or col.strip() == "*":
#             expr = f"SUM({weight_expr})"
#         else:
#             expr = f"SUM(CASE WHEN ({col}) IS NOT NULL THEN {weight_expr} ELSE 0 END)"
#     elif fn == "avg":
#         expr = f"SUM(({col}) * {weight_expr}) / NULLIF(SUM({weight_expr}), 0)"
#     else:
#         raise ValueError(f"unsupported aggregate {agg.func}")
#     if alias:
#         return f"{expr} AS {alias}"
#     return expr


def rewrite_query_with_plan(parse_result, plan: SamplePlan) -> Tuple[str, List[str]]:
    notes: List[str] = []
    from_parts: List[str] = []
    alias_map = {}  # original table alias -> sampled alias name

    def safe_alias(name: str) -> str:
        return re.sub(r'[^0-9A-Za-z_]', '_', name) + "__s"

    for t in parse_result.tables:
        key = t.alias or t.name
        entry = plan.entries.get(key)
        if entry and entry.subquery_sql:
            # entry.subquery_sql already contains TABLESAMPLE and weight column
            from_parts.append(entry.subquery_sql)
            # sampled alias as produced by build_sample_subquery: <tablealias>__s
            # sampled_alias = (t.alias or re.sub(r'[^0-9A-Za-z_]', '_', t.name)) + "__s"
            sampled_alias = safe_alias(key)
            alias_map[key] = sampled_alias
        elif entry and entry.method == 'full':
            qualified = f"{t.name} AS {key}"
            from_parts.append(qualified)
            alias_map[key] = key
            notes.append(f"No sampling for {key}: using full table")
        else:
            qualified = f"{t.name} AS {key}"
            from_parts.append(qualified)
            alias_map[key] = key
            notes.append(f"No plan entry for {key}; using full table")

    select_items = []
    group_items = []
    if parse_result.group_by:
        if len(parse_result.tables) == 1:
            tbl = parse_result.tables[0]
            tbl_key = tbl.alias or tbl.name
            sampled_alias = alias_map.get(tbl_key, tbl_key)
            for g in parse_result.group_by:
                if '.' in g or '(' in g:
                    group_items.append(g)
                    select_items.append(g)
                else:
                    group_items.append(f"{sampled_alias}.{g}")
                    select_items.append(f"{sampled_alias}.{g}")
        else:
            for g in parse_result.group_by:
                group_items.append(g)
                select_items.append(g)
    for agg in parse_result.aggregates:
        # heuristics to pick table alias for the aggregate: if only one table, use it; else use first matching column.table
        if len(parse_result.tables) == 1:
            tbl = parse_result.tables[0]
            tbl_key = tbl.alias or tbl.name
            sampled_alias = alias_map.get(tbl_key, tbl_key)
            weight_col = f"{sampled_alias}._sample_weight"
            if agg.func.lower() == 'avg':
                expr = f"SUM(({sampled_alias}.{agg.column}) * {weight_col}) / NULLIF(SUM({weight_col}), 0)"
            elif agg.func.lower() == 'sum':
                expr = f"SUM(({sampled_alias}.{agg.column}) * {weight_col})"
            elif agg.func.lower() == 'count':
                if not agg.column or agg.column.strip() == '*':
                    expr = f"SUM({weight_col})"
                else:
                    expr = f"SUM(CASE WHEN ({sampled_alias}.{agg.column}) IS NOT NULL THEN {weight_col} ELSE 0 END)"
            else:
                raise ValueError(f"unsupported aggregate: {agg.func}")
            alias_name = getattr(agg, 'alias', None) or f"agg_{len(select_items)}"
            select_items.append(f"{expr} AS {alias_name}")
        else:
            # For multi-table aggregates you need join-aware logic. Fallback heuristic:
            tbl = parse_result.tables[0]
            tbl_key = tbl.alias or tbl.name
            sampled_alias = alias_map.get(tbl_key, tbl_key)
            weight_col = f"{sampled_alias}._sample_weight"
            expr = f"SUM(({sampled_alias}.{agg.column}) * {weight_col})"
            alias_name = getattr(agg, 'alias', None) or f"agg_{len(select_items)}"
            select_items.append(f"{expr} AS {alias_name}")

    select_clause = ", ".join(select_items)
    from_clause = ", ".join(from_parts)
    group_clause = f" GROUP BY {', '.join(group_items)}" if group_items else ""
    sql = f"SELECT {select_clause} FROM {from_clause}{group_clause};"
    return sql, notes



def execute_plan_and_confidence(parse_result: ParseResult,
                                plan: SamplePlan,
                                db_execute: Callable[[str], dict],
                                p_confidence: float) -> Tuple[dict, List[str]]:

    sql, notes = rewrite_query_with_plan(parse_result, plan)
    res = db_execute(sql)
    rows = res.get("rows", [])
    if not rows:
        raise RuntimeError("final sampled SQL returned no rows")

    if parse_result.group_by:
        cols = res.get("columns", [])
        estimate = rows
        return ({"estimate": estimate, "columns": cols, "ci_lower": None, "ci_upper": None, "estimated_variance": None, "p_used": p_confidence}, notes)
    first_row = rows[0]
    if isinstance(first_row, (list, tuple)):
        values = list(first_row)
        estimate = values[0] if len(values) == 1 else values
    else:
        estimate = first_row

    combined_uv = 0.0
    has_uv = False
    for k, entry in plan.entries.items():
        if getattr(entry, "chosen_uv", None) is not None:
            combined_uv += entry.chosen_uv or 0.0
            has_uv = True

    if not has_uv:
        notes.append("No U_V[Theta] available from plan entries; cannot compute proper CI. Returning estimate without CI.")
        return ({"estimate": estimate, "ci_lower": None, "ci_upper": None, "estimated_variance": None, "p_used": p_confidence}, notes)

    from utils.bsap import z_from_quantile
    quantile = (1.0 + p_confidence) / 2.0
    z = z_from_quantile(quantile)

    se = math.sqrt(max(0.0, float(combined_uv)))
    if isinstance(estimate, list):
        
        ci_lower = [float(val) - z * se for val in estimate]
        ci_upper = [float(val) + z * se for val in estimate]
    else:
        ci_lower = float(estimate) - z * se
        ci_upper = float(estimate) + z * se

    return ({"estimate": estimate, "ci_lower": ci_lower, "ci_upper": ci_upper,
             "estimated_variance": combined_uv, "p_used": p_confidence}, notes)