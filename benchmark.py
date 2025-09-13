# benchmark.py
import time
import statistics
import json
from typing import Any, Dict, Tuple, List, Optional

from db_connect import PSQL, QQLEnv
from execute import parser, execute_query   # execute_query uses online handler and exact fallback
import online_query_handler

# Helper: run and time a callable
def time_call(fn, *args, **kwargs) -> Tuple[float, Any]:
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    t1 = time.perf_counter()
    return (t1 - t0), out

# Helper: obtain DB-side execution time using EXPLAIN ANALYZE
def db_execution_time(db: PSQL, sql: str) -> Optional[float]:
    """
    Returns the execution time in milliseconds parsed from EXPLAIN ANALYZE output,
    or None if it cannot be parsed.

    WARNING: EXPLAIN ANALYZE will actually execute the query.
    """
    try:
        cur = db.cur
        cur.execute(f"EXPLAIN ANALYZE {sql}")
        rows = cur.fetchall()
        text = "\n".join(r[0] for r in rows)
        # Look for line with 'Execution Time: <ms> ms'
        for line in text.splitlines()[::-1]:
            line = line.strip()
            if line.lower().startswith("execution time:") and "ms" in line.lower():
                # example: Execution Time: 240.123 ms
                try:
                    parts = line.split(":")[1].strip().split()
                    ms = float(parts[0])
                    return ms
                except Exception:
                    continue
        # fallback: no parseable execution time
        return None
    except Exception as e:
        print("db_execution_time error:", e)
        return None

# Run one exact execution and measure time-to-output (wall clock) and DB-side (explain analyze)
def run_exact_once(raw_sql: str, db: PSQL, explain_db: bool = False) -> Dict[str, Any]:
    pr = parser.parse(raw_sql)
    # cleaned_sql is what we execute exactly
    cleaned = pr.cleaned_sql
    # time wall-clock exact execution (use db.execute which returns cursor)
    t_wall, cur = time_call(db.execute, cleaned)
    # fetch the result (cursor is returned)
    # ensure we materialize results so timing is end-to-end
    try:
        rows = cur.fetchall()
    except Exception:
        # db.execute in your wrapper returns cursor already fetched?? attempt to re-query
        rows = []
    db_ms = None
    if explain_db:
        db_ms = db_execution_time(db, cleaned)  # in ms
    return {"mode": "exact", "wall_seconds": t_wall, "db_ms": db_ms, "rows": rows, "cleaned_sql": cleaned}

# Run one approx execution and measure wall-clock (end-to-end) + optionally DB explain analyze of final SQL
def run_approx_once(raw_sql: str, db: PSQL, explain_db: bool = False) -> Dict[str, Any]:
    # Use your execute_query wrapper which calls online handler and returns structured result
    t_wall, result = time_call(execute_query, raw_sql, db)
    # execute_query returns structured dict if approx succeeded, or fallback exact
    db_ms = None
    final_sql = result.get("final_sql")
    if explain_db and final_sql:
        # final_sql may be the rewritten sampled SQL; explain it
        db_ms = db_execution_time(db, final_sql)
    return {"mode": "approx", "wall_seconds": t_wall, "db_ms": db_ms, "result": result, "final_sql": final_sql}

# Build a simple harness to run multiple trials, warmups, and summarize
def run_benchmark(raw_sql: str, db: PSQL, runs: int = 5, warmup: int = 1, explain_db: bool = False) -> Dict[str, Any]:
    exact_times = []
    approx_times = []
    exact_db_ms = []
    approx_db_ms = []
    exact_rows = None
    approx_results = []

    # Warmup: run once exact & approx to warm caches if requested
    if warmup:
        print("Warmup exact...")
        try:
            run_exact_once(raw_sql, db, explain_db=False)
        except Exception as e:
            print("Warmup exact error:", e)
        print("Warmup approx...")
        try:
            run_approx_once(raw_sql, db, explain_db=False)
        except Exception as e:
            print("Warmup approx error:", e)

    for i in range(runs):
        print(f"=== Trial {i+1}/{runs} ===")
        e = run_exact_once(raw_sql, db, explain_db=explain_db)
        exact_times.append(e["wall_seconds"])
        if explain_db:
            exact_db_ms.append(e["db_ms"])
        if exact_rows is None:
            exact_rows = e["rows"]

        a = run_approx_once(raw_sql, db, explain_db=explain_db)
        approx_times.append(a["wall_seconds"])
        approx_results.append(a)
        if explain_db:
            approx_db_ms.append(a["db_ms"])

        print(f"Exact wall: {e['wall_seconds']:.4f}s, Approx wall: {a['wall_seconds']:.4f}s")
        if explain_db:
            print(f"Exact DB ms: {e['db_ms']}, Approx DB ms: {a['db_ms']}")
        # optionally small sleep between runs
        time.sleep(0.1)

    def summarize(name: str, lst: List[float]) -> Dict[str, float]:
        return {
            "count": len(lst),
            "mean": statistics.mean(lst) if lst else float("nan"),
            "median": statistics.median(lst) if lst else float("nan"),
            "stdev": statistics.stdev(lst) if len(lst) > 1 else 0.0,
        }

    summary = {
        "sql": raw_sql,
        "exact": summarize("exact", exact_times),
        "approx": summarize("approx", approx_times),
        "exact_db_ms": {"median": statistics.median([x for x in exact_db_ms if x is not None]) if exact_db_ms else None},
        "approx_db_ms": {"median": statistics.median([x for x in approx_db_ms if x is not None]) if approx_db_ms else None},
        "trials": runs,
        "warmup_runs": warmup,
        "approx_details": approx_results,
        "exact_rows_sample": exact_rows[:5] if exact_rows else None,
    }

    # compute overall speedup
    try:
        summary["speedup_median"] = summary["exact"]["median"] / summary["approx"]["median"]
    except Exception:
        summary["speedup_median"] = None

    return summary

# Example usage
if __name__ == "__main__":
    env = QQLEnv()
    db = PSQL(dbname='postgres' , 
          user='postgres.yvaqsuzfmizddzkeksjy', 
          password='qql1234', 
          host='aws-1-ap-south-1.pooler.supabase.com', 
          port='6543')
  
    sql = "SELECT APPROX SUM(price) FROM pizza_orders ERROR 0.03 PROB 0.98"
    res = run_benchmark(sql, db, runs=3, warmup=1, explain_db=True)
    print(json.dumps(res, indent=2, default=str))
    db.close()
