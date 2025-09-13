# rewriter.py
from typing import Tuple, List, Callable
from online_query_sampler import SamplePlan, SamplePlanEntry, build_sample_subquery
from parser import ParseResult, AggregateInfo
import math
import re

def _rewrite_agg_weighted(agg: AggregateInfo, weight_expr: str):
    fn = agg.func.lower()
    col = agg.column or ''
    alias = getattr(agg, "alias", None) or None
    if fn == "sum":
        expr = f"SUM(({col}) * {weight_expr})"
    elif fn == "count":
        if not col or col.strip() == "*":
            expr = f"SUM({weight_expr})"
        else:
            expr = f"SUM(CASE WHEN ({col}) IS NOT NULL THEN {weight_expr} ELSE 0 END)"
    elif fn == "avg":
        expr = f"SUM(({col}) * {weight_expr}) / NULLIF(SUM({weight_expr}), 0)"
    else:
        raise ValueError(f"unsupported aggregate {agg.func}")
    if alias:
        return f"{expr} AS {alias}"
    return expr


def rewrite_query_with_plan(parse_result, plan: SamplePlan) -> Tuple[str, List[str]]:
    notes: List[str] = []
    from_parts: List[str] = []
    alias_map = {}  # original table alias -> sampled alias name

    for t in parse_result.tables:
        key = t.alias or t.name
        entry = plan.entries.get(key)
        if entry and entry.subquery_sql:
            # entry.subquery_sql already contains TABLESAMPLE and weight column
            from_parts.append(entry.subquery_sql)
            # sampled alias as produced by build_sample_subquery: <tablealias>__s
            sampled_alias = (t.alias or re.sub(r'[^0-9A-Za-z_]', '_', t.name)) + "__s"
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

    # Build SELECT list using sampled alias and weight column
    select_items = []
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
    # If WHERE exists and you didn't push it into table.where earlier, ensure it's applied inside sampled subqueries.
    # Our build_sample_subquery reads table.where — ensure the parser.enrich pushed parse_result.where into table.where for single-table queries.
    sql = f"SELECT {select_clause} FROM {from_clause};"
    return sql, notes



def execute_plan_and_confidence(parse_result: ParseResult,
                                plan: SamplePlan,
                                db_execute: Callable[[str], dict],
                                p_confidence: float) -> Tuple[dict, List[str]]:
    """
    Run the rewritten sampled SQL and return an approximate answer and confidence interval.

    Returns (result_dict, notes) where result_dict includes keys:
      - 'estimate': numeric (point estimate)
      - 'ci_lower', 'ci_upper': confidence interval bounds (two-sided) for the requested p_confidence
      - 'estimated_variance': U_V[Theta] used for the confidence interval (if available)
      - 'p_used': p_confidence
      - 'agg_aliases': list of aggregates order corresponding to estimate components (if multiple)
    """

    sql, notes = rewrite_query_with_plan(parse_result, plan)
    res = db_execute(sql)
    rows = res.get("rows", [])
    if not rows:
        raise RuntimeError("final sampled SQL returned no rows")

    # For simplicity we expect a single row containing the aggregate(s)
    first_row = rows[0]
    # If only one aggregate, first_row may be a singleton
    if isinstance(first_row, (list, tuple)):
        values = list(first_row)
        estimate = values[0] if len(values) == 1 else values
    else:
        estimate = first_row

    # Determine estimated variance U_V[Theta] for CI. If plan entries contain chosen_uv combine them.
    # For single-table single-aggregate scenario: use that chosen_uv
    combined_uv = 0.0
    has_uv = False
    for k, entry in plan.entries.items():
        if getattr(entry, "chosen_uv", None) is not None:
            combined_uv += entry.chosen_uv or 0.0
            has_uv = True

    if not has_uv:
        notes.append("No U_V[Theta] available from plan entries; cannot compute proper CI. Returning estimate without CI.")
        return ({"estimate": estimate, "ci_lower": None, "ci_upper": None, "estimated_variance": None, "p_used": p_confidence}, notes)

    # Two-sided z quantile
    from bsap import z_from_quantile
    quantile = (1.0 + p_confidence) / 2.0
    z = z_from_quantile(quantile)

    # For the inequality used in Procedure1 the random variable is approx normal with var = combined_uv.
    # We produce a two-sided CI: estimate +/- z * sqrt(UV)
    se = math.sqrt(max(0.0, float(combined_uv)))
    if isinstance(estimate, list):
        # if multiple aggregates we conservatively use the same combined_uv for each
        ci_lower = [float(val) - z * se for val in estimate]
        ci_upper = [float(val) + z * se for val in estimate]
    else:
        ci_lower = float(estimate) - z * se
        ci_upper = float(estimate) + z * se

    return ({"estimate": estimate, "ci_lower": ci_lower, "ci_upper": ci_upper,
             "estimated_variance": combined_uv, "p_used": p_confidence}, notes)


# Demo harness using mock DB (if run directly) --------------------------------
if __name__ == "__main__":
    # For demo purposes we show a minimal flow: create fake parse_result & plan,
    # rewrite and execute using a mock db_execute that computes the weighted sum exactly.

    from sampler import TableInfo, AggregateInfo, ParseResult
    from taqa import taqa_procedure1

    # create a ParseResult with one table and one aggregate
    pr = ParseResult()
    pr.tables = [TableInfo(name="orders", alias="o")]
    pr.tables[0].num_blocks = 1000  # pretend we have 1000 blocks
    a = AggregateInfo(func="sum", column="amount", raw_ast=None)
    a.alias = "sum_amount"
    pr.aggregates = [a]

    # A mock db_execute that can answer both pilot block aggregation requests and
    # final sampled queries.
    import random
    # Simulate a population of B blocks with block totals
    B = 1000
    # create synthetic block totals (true population block sums)
    true_block_sums = [random.expovariate(1 / 1000.0) for _ in range(B)]
    true_total = sum(true_block_sums)

    def mock_db_execute(sql: str) -> dict:
        # If PILOT marker in sql, return pilot block sums
        if "PILOT BLOCK SAMPLE" in sql:
            # parse pilot_frac_blocks from the SQL comment
            import re
            m = re.search(r"pilot_frac_blocks=([0-9.]+)", sql)
            pf = float(m.group(1)) if m else 0.01
            pilot_blocks = max(1, int(pf * B))
            sampled = random.sample(true_block_sums, pilot_blocks)
            return {"sample_stats": {"sum_amount": {"block_sums": sampled}}}
        else:
            # Final sampled query: we assume the query uses TABLESAMPLE SYSTEM to pick some blocks,
            # but to keep the mock simple accept sample_frac in the SQL string
            m = re.search(r"TABLESAMPLE SYSTEM \\(([^\\)]+)\\)", sql)
            if m:
                pct = float(m.group(1))
                frac = pct / 100.0
            else:
                # fallback: use fraction 1.0
                frac = 1.0
            # sample approx frac * B blocks
            sample_n = max(1, int(frac * B))
            sampled = random.sample(true_block_sums, sample_n)
            # If the rewritten SQL used an HT weight (1/frac) inside SUM, then the estimate is:
            # estimate = sum(sampled_block_sums) * (1/frac)  [since block sums were summed]
            est = sum(sampled) * (1.0 / max(1e-12, frac))
            # return as rows with single value
            return {"rows": [(est,)]}

    # Run TAQA procedure to get a plan
    plan, notes = taqa_procedure1(pr, eps=0.05, p_confidence=0.95, db_execute=mock_db_execute, theta_p=0.01)
    print("TAQA notes:")
    print("\n".join(notes))
    print("Plan:", plan.entries)

    # Build and run final query + CI
    result, rnotes = execute_plan_and_confidence(pr, plan, mock_db_execute, p_confidence=0.95)
    print("Result:", result)
    print("Notes:", rnotes)
