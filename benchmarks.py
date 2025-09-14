import time
from statistics import mean
from typing import List, Tuple, Dict, Any
from decimal import Decimal

from utils.db_connect import PSQL, QQLEnv
from execute import execute_query

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import math


def _to_float(x):
    if isinstance(x, Decimal):
        return float(x)
    try:
        return float(x)
    except Exception:
        return None


def run_once_exact(db: PSQL, sql_exact: str) -> Tuple[float, float]:
    t0 = time.perf_counter()
    cur = db.execute(sql_exact)
    row = cur.fetchone()
    t1 = time.perf_counter()

    val = None
    if row is not None:
        v = row[0]
        val = _to_float(v)
    return val, (t1 - t0)


def run_once_approx(db: PSQL, sql_approx: str) -> Tuple[float, float]:
    t0 = time.perf_counter()
    res = execute_query(sql_approx, db)
    t1 = time.perf_counter()
    est = res.get("estimate", {}).get("estimate") if isinstance(res, dict) else None
    val = _to_float(est)
    return val, (t1 - t0)


def benchmark_accuracy_speed(db: PSQL,
                             base_sql: str,
                             agg_exact_sql: str,
                             errors: List[float],
                             trials: int = 3) -> Dict[str, Any]:
    exact_val, exact_time = run_once_exact(db, agg_exact_sql)

    approx_points = []
    for eps in errors:
        approx_sql = base_sql.replace("ERROR 0.05", f"ERROR {eps}")
        est_vals = []
        times = []
        for _ in range(trials):
            v, dt = run_once_approx(db, approx_sql)
            if v is not None:
                est_vals.append(v)
                times.append(dt)
        if est_vals:
            avg_est = mean(est_vals)
            avg_time = mean(times)
            err = abs(avg_est - exact_val) if exact_val is not None else None
            approx_points.append({
                "eps": eps,
                "estimate": avg_est,
                "time": avg_time,
                "abs_error": err
            })

    return {
        "exact": {"value": exact_val, "time": exact_time},
        "approx": approx_points
    }


def plot_benchmarks(result: Dict[str, Any], out_path: str):
    exact = result["exact"]
    approx = result["approx"]

    approx_sorted = sorted(approx, key=lambda p: p["eps"]) if approx else []
    xs = [p["eps"] for p in approx_sorted]
    times = [p["time"] for p in approx_sorted]
    errors = [p["abs_error"] for p in approx_sorted]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(xs, times, marker='o')
    axes[0].axhline(y=exact["time"], color='r', linestyle='--')
    axes[0].set_xlabel("ERROR (eps)")
    axes[0].set_ylabel("Time (s)")
    axes[0].set_title("Speed: Approx vs Exact")

    axes[1].plot(xs, errors, marker='o')
    axes[1].set_xlabel("ERROR (eps)")
    axes[1].set_ylabel("Absolute Error vs Exact")
    axes[1].set_title("Accuracy: Approx vs Exact")

    if xs:
        start = math.floor(min(xs) / 0.05) * 0.05
        end = math.ceil(max(xs) / 0.05) * 0.05
        steps = int(round((end - start) / 0.05))
        ticks = [round(start + i * 0.05, 2) for i in range(steps + 1)]
        axes[0].set_xticks(ticks)
        axes[1].set_xticks(ticks)

    fig.tight_layout()
    fig.savefig(out_path)


if __name__ == "__main__":
    env = QQLEnv()
    db = PSQL(
        dbname=env.dbname,
        user=env.username,
        password=env.password,
        host=env.host,
        port=str(env.port),
    )

    errors = [round(i * 0.05, 2) for i in range(1, 13)]

    import os
    os.makedirs("generated", exist_ok=True)

    suites = [
        (
            "avg",
            "SELECT APPROX AVG(price) FROM pizza_orders ERROR 0.05 PROB 0.98;",
            "SELECT AVG(price) FROM pizza_orders;",
        ),
        (
            "sum",
            "SELECT APPROX SUM(price) FROM pizza_orders ERROR 0.05 PROB 0.98;",
            "SELECT SUM(price) FROM pizza_orders;",
        ),
        (
            "count",
            "SELECT APPROX COUNT(price) FROM pizza_orders ERROR 0.05 PROB 0.98;",
            "SELECT COUNT(price) FROM pizza_orders;",
        ),
    ]

    for label, approx_base, exact_sql in suites:
        res = benchmark_accuracy_speed(db, approx_base, exact_sql, errors, trials=3)
        out_png = f"generated/benchmark_accuracy_speed_{label}.png"
        plot_benchmarks(res, out_png)
        print({
            "agg": label,
            "exact": res["exact"],
            "approx": res["approx"],
            "plot": out_png,
        })

    db.close()
