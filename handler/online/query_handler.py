import re
from typing import Dict, Any, Callable

from utils.taqa import taqa_procedure1
from handler.online.rewriter import rewrite_query_with_plan, execute_plan_and_confidence

from utils.db_connect import PSQL

_PILOT_FRAC_RE = re.compile(r"pilot_frac_blocks=([0-9]*\.?[0-9]+)")

def _make_db_execute(db: PSQL, parse_result) -> Callable[[str], Dict[str, Any]]:
    def db_execute(sql: str) -> Dict[str, Any]:
        cur = db.cur
        sql_lower = sql.lower()
        if "pilot block sample" in sql_lower or "pilot_frac_blocks" in sql_lower:
            m = _PILOT_FRAC_RE.search(sql)
            pilot_frac = float(m.group(1)) if m else 0.01
            sample_stats = {}
            for tbl in parse_result.tables:
                table_name = tbl.name
                where_clause = f"WHERE {parse_result.where}" if getattr(parse_result, "where", None) else ""
                percent = pilot_frac * 100.0
                for agg in parse_result.aggregates:
                    agg_col = agg.column
                    func_lower = agg.func.lower()
                    if func_lower == 'avg':
                        pilot_block_sql = (
                            f"SELECT page_id, SUM({agg_col}) AS block_sum, COUNT({agg_col}) AS block_count FROM (\n"
                            f"  SELECT (t.ctid::text::point)[0]::int AS page_id, {agg_col}\n"
                            f"  FROM {table_name} AS t TABLESAMPLE SYSTEM ({percent}) {where_clause}\n"
                            f") s GROUP BY page_id"
                        )
                        cur.execute(pilot_block_sql)
                        rows = cur.fetchall()
                        block_sums = [row[1] for row in rows]
                        block_counts = [row[2] for row in rows]
                        n_blocks = len(block_sums)
                        key_alias = getattr(agg, "alias", None)
                        keys = []
                        if key_alias:
                            keys.append(key_alias)
                        keys.append(f"{agg.func}({agg.column})")
                        keys.append(f"{agg.func.lower()}({agg.column})")
                        for k in keys:
                            sample_stats.setdefault(k, {})
                            sample_stats[k]["block_sums"] = block_sums
                            sample_stats[k]["block_counts"] = block_counts
                            sample_stats[k]["n_blocks"] = n_blocks
                        continue
                    pilot_block_sql = (
                        f"SELECT page_id, SUM({agg_col}) AS block_sum FROM (\n"
                        f"  SELECT (t.ctid::text::point)[0]::int AS page_id, {agg_col}\n"
                        f"  FROM {table_name} AS t TABLESAMPLE SYSTEM ({percent}) {where_clause}\n"
                        f") s GROUP BY page_id"
                    )
                    cur.execute(pilot_block_sql)
                    rows = cur.fetchall()
                    block_sums = [row[1] for row in rows]
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
            cur.execute(sql)
            rows = cur.fetchall()
            return {"rows": rows}
    return db_execute

def handle_online(parse_result, db: PSQL) -> Dict[str, Any]:
    result = {"success": False, "meta": {}, "estimate": None}
    db_execute = _make_db_execute(db, parse_result)
    try:
        plan, taqa_notes = taqa_procedure1(parse_result, eps=parse_result.error,
                                           p_confidence=parse_result.confidence,
                                           db_execute=db_execute,
                                           theta_p=0.01)
    except Exception as e:
        return {"success": False, "error": str(e), "meta": {"note": "TAQA failed"}}
    if not plan.entries:
        return {"success": False, "error": "No sample plan produced", "meta": {"notes": taqa_notes}}
    final_sql, rewrite_notes = rewrite_query_with_plan(parse_result, plan)
    try:
        exec_result, exec_notes = execute_plan_and_confidence(parse_result, plan, db_execute, p_confidence=parse_result.confidence)
    except Exception as e:
        return {"success": False, "error": str(e), "meta": {"taqa": taqa_notes, "rewrite": rewrite_notes}}
    return {
        "success": True,
        "mode": "online",
        "final_sql": final_sql,
        "estimate": exec_result,
        "taqa_notes": taqa_notes,
        "rewrite_notes": rewrite_notes + exec_notes
    }
