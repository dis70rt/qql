# online_query_handler.py
import re
import math
from typing import Dict, Any, Callable, Tuple, List

from online_query_sampler import SamplePlan, SamplePlanEntry, build_sample_subquery
from taqa import taqa_procedure1
from online_rewriter import rewrite_query_with_plan, execute_plan_and_confidence

# PSQL is your DB wrapper from db_connect.py
from db_connect import PSQL

# Helper: extract pilot_frac from the pilot SQL comment that taqa_procedure1 emits.
_PILOT_FRAC_RE = re.compile(r"pilot_frac_blocks=([0-9]*\.?[0-9]+)")

def _make_db_execute(db: PSQL, parse_result) -> Callable[[str], Dict[str, Any]]:
    """
    Return a db_execute(sql) closure that:
     - when called with a pilot SQL (taqa_procedure1 emits a SQL containing the marker
       string 'PILOT BLOCK SAMPLE' or pilot_frac_blocks comment), it will execute
       a block-aggregation pilot using GROUP BY page_id (using ctid).
     - otherwise, it runs the SQL as normal and returns {'rows': cursor.fetchall()}.
     
    The closure has access to parse_result so it can compute the correct pilot
    block-aggregation for the aggregates and tables referenced by the query.
    """
    def db_execute(sql: str) -> Dict[str, Any]:
        cur = db.cur
        sql_lower = sql.lower()
        # Detect pilot marker (we used a comment string in taqa_procedure1)
        if "pilot block sample" in sql_lower or "pilot_frac_blocks" in sql_lower:
            # parse pilot_frac_blocks if included
            m = _PILOT_FRAC_RE.search(sql)
            pilot_frac = float(m.group(1)) if m else 0.01
            sample_stats = {}
            # For each aggregate, compute the block sums via a GROUP BY page_id
            # We'll group inside a subquery that samples blocks via TABLESAMPLE SYSTEM
            for tbl in parse_result.tables:
                table_name = tbl.name
                # push parse-level where predicate into table-level where if present
                where_clause = f"WHERE {parse_result.where}" if getattr(parse_result, "where", None) else ""
                # build sampled inner select: extract page_id via ctid trick
                percent = pilot_frac * 100.0
                inner = (
                    f"SELECT (t.ctid::text::point)[0]::int AS page_id, "
                    # we will compute block sums per each aggregate column later
                    f"t.* FROM {table_name} AS t TABLESAMPLE SYSTEM ({percent}) {where_clause}"
                )
                # For each aggregate, compute block sums (we use parse_result.aggregates)
                for agg in parse_result.aggregates:
                    # figure aggregate column (string like 'amount')
                    agg_col = agg.column
                    # build SQL to compute block sums for this aggregate on this table
                    # Only compute if agg likely references this table (simple heuristic)
                    # If aggregate references other table, skip — taqa_procedure1 selected candidate table heuristically
                    pilot_block_sql = (
                        f"SELECT page_id, SUM({agg_col}) AS block_sum FROM (\n"
                        f"  SELECT (t.ctid::text::point)[0]::int AS page_id, {agg_col}\n"
                        f"  FROM {table_name} AS t TABLESAMPLE SYSTEM ({percent}) {where_clause}\n"
                        f") s GROUP BY page_id"
                    )
                    cur.execute(pilot_block_sql)
                    rows = cur.fetchall()
                    block_sums = [row[1] for row in rows]
                    # compute basic stats (mean/var/n) to return; taqa uses block_sums if present
                    n_blocks = len(block_sums)
                    if n_blocks == 0:
                        block_mean = 0.0
                        block_var = 0.0
                    else:
                        block_mean = sum(block_sums) / n_blocks
                        if n_blocks > 1:
                            block_var = sum((x - block_mean) ** 2 for x in block_sums) / (n_blocks - 1)
                        else:
                            block_var = 0.0
                    # construct keys taqa expects: try agg.alias first then sensible signatures
                    key_alias = getattr(agg, "alias", None)
                    keys = []
                    if key_alias:
                        keys.append(key_alias)
                    keys.append(f"{agg.func}({agg.column})")
                    keys.append(f"{agg.func.lower()}({agg.column})")
                    for k in keys:
                        sample_stats.setdefault(k, {})
                        sample_stats[k]["block_sums"] = block_sums
                        sample_stats[k]["block_mean"] = block_mean
                        sample_stats[k]["block_var"] = block_var
                        sample_stats[k]["n_blocks"] = n_blocks
            return {"sample_stats": sample_stats}
        else:
            # Final (non-pilot) SQL execution
            cur.execute(sql)
            rows = cur.fetchall()
            return {"rows": rows}
    return db_execute


def handle_online(parse_result, db: PSQL) -> Dict[str, Any]:
    """
    High-level online handler:
      1. Run TAQA (Procedure 1) using db_execute closure
      2. Rewrite query using returned SamplePlan
      3. Execute final sampled SQL, compute CI and return results
    """
    result = {"success": False, "meta": {}, "estimate": None}

    db_execute = _make_db_execute(db, parse_result)

    # Run TAQA (Procedure 1). It will call db_execute for pilot queries.
    try:
        plan, taqa_notes = taqa_procedure1(parse_result, eps=parse_result.error,
                                           p_confidence=parse_result.confidence,
                                           db_execute=db_execute,
                                           theta_p=0.01)
    except Exception as e:
        return {"success": False, "error": str(e), "meta": {"note": "TAQA failed"}}

    # If TAQA returns empty plan entries, fallback to exact
    if not plan.entries:
        return {"success": False, "error": "No sample plan produced", "meta": {"notes": taqa_notes}}

    # Build final SQL
    final_sql, rewrite_notes = rewrite_query_with_plan(parse_result, plan)

    # Execute final SQL and compute CI (execute_plan_and_confidence expects a db_execute that returns 'rows')
    try:
        exec_result, exec_notes = execute_plan_and_confidence(parse_result, plan, db_execute, p_confidence=parse_result.confidence)
    except Exception as e:
        return {"success": False, "error": str(e), "meta": {"taqa": taqa_notes, "rewrite": rewrite_notes}}

    # Return structured response
    return {
        "success": True,
        "mode": "online",
        "final_sql": final_sql,
        "estimate": exec_result,
        "taqa_notes": taqa_notes,
        "rewrite_notes": rewrite_notes + exec_notes
    }
