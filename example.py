# run_aqe_example.py
import re
import random

# Adjust these imports to your project layout if needed:
# - parser.AQE_Parser should be the parser you provided earlier
# - taqa.taqa_procedure1 is the Procedure 1 implementation for TAQA
# - rewriter.rewrite_query_with_plan produces the final SQL from the SamplePlan
from parser import AQE_Parser
from taqa import taqa_procedure1
from online_rewriter import rewrite_query_with_plan
from online_query_sampler import TableInfo, AggregateInfo

SQL = """SELECT APPROX SUM(amount)
FROM orders
ERROR 0.15 PROB 0.85"""

def make_parse_result(sql_text):
    """
    Parse the SQL using your AQE_Parser and then *augment* table metadata
    so TAQA can run (we need num_blocks to enable block-TAQA in this demo).
    When running for real, num_blocks should be discovered/estimated from DB statistics.
    """
    parser = AQE_Parser()
    pr = parser.parse(sql_text)

    # For this demo we must ensure table objects have num_blocks.
    # If your ParseResult.tables entries are parser.TableInfo objects,
    # we set a synthetic num_blocks for the largest table (e.g., orders).
    if pr.tables:
        # assign num_blocks heuristically for demo
        for t in pr.tables:
            # choose a synthetic block count; in real runs read from DB metadata
            if t.alias == 'o' or 'order' in t.name.lower():
                t.num_blocks = 1000   # pretend orders has 1000 blocks
            else:
                t.num_blocks = 800    # customers 800 blocks
    # Also ensure aggregates have aliases so TAQA can index them
    for i, agg in enumerate(pr.aggregates):
        if getattr(agg, 'raw_ast', None) is not None and not getattr(agg, 'column', None):
            pass
        if not getattr(agg, 'alias', None):
            # set a stable alias like sum_amount
            agg.alias = f"agg_{i}"
    return pr

def mock_db_execute(sql: str) -> dict:
    """
    Mock db_execute used by TAQA in the demo.
    It recognizes pilot SQL (our taqa_procedure1 emits a pilot_sql comment)
    and returns sample_stats with block_sums so TAQA can compute L_mu and choose n.
    For final sampled SQL (rewriter output), we simply return example rows.
    """
    # Detect pilot block sample (our taqa_procedure1 uses a comment hack)
    if "PILOT BLOCK SAMPLE" in sql or "pilot_frac_blocks" in sql:
        # extract pilot_frac_blocks from the comment if present
        m = re.search(r"pilot_frac_blocks=([0-9.]+)", sql)
        pilot_frac = float(m.group(1)) if m else 0.01
        # determine pilot_blocks (simulate using 1000 blocks for orders example)
        # choose same B as in parse_result augmentation in demo
        B = 1000
        pilot_blocks = max(1, int(B * pilot_frac))
        # produce synthetic block sums that give a non-zero mean but small variance
        # (so TAQA will quickly find a small n)
        # e.g., block sums around 1000 with tiny noise
        block_sums = [1000.0 + random.uniform(-1, 1) for _ in range(pilot_blocks)]
        # the aggregate key must match the alias TAQA expects
        # TAQA uses agg.alias or f"{agg.func}({agg.column})"
        # For our parser we set aliases to agg_0 etc; but to be robust return for both keys:
        return {
            "sample_stats": {
                # example keys TAQA might look for:
                "agg_0": {"block_sums": block_sums},
                "SUM(amount)": {"block_sums": block_sums},
                "sum(amount)": {"block_sums": block_sums},
            }
        }

    # Otherwise it's the final rewritten SQL produced by rewriter.
    # The rewriter will generate SQL that uses TABLESAMPLE SYSTEM and weights.
    # To simulate, detect TABLESAMPLE percent and compute estimate from synthetic population.
    m_pct = re.search(r"TABLESAMPLE SYSTEM \(([0-9.]+)\)", sql, re.IGNORECASE)
    if m_pct:
        pct = float(m_pct.group(1)) / 100.0
    else:
        # If not present, assume full scan
        pct = 1.0

    # For the demo we assume true population sum = 1e6 (just a synthetic number)
    true_sum = 1_000_000.0
    # sample estimate under HT = (sum of sampled_block_sums) * (1/pct)
    # simulate sampling noise:
    sampled_fraction = max(0.001, pct)
    # a simple noisy estimate
    sampled_sum = true_sum * sampled_fraction * (1.0 + random.uniform(-0.02, 0.02))
    estimate = sampled_sum / max(1e-12, sampled_fraction)   # HT correction
    # return as rows (tuple)
    return {"rows": [(estimate,)]}


def main():
    # 1) parse
    pr = make_parse_result(SQL)
    print("Parsed. Plan mode:", pr.plan_mode)
    print("Tables:", [(t.name, t.alias, getattr(t, "num_blocks", None)) for t in pr.tables])
    print("Aggregates:", [(a.func, a.column, a.alias) for a in pr.aggregates])

    # 2) run TAQA (Procedure 1) to get SamplePlan
    eps = pr.error   # 0.03
    p_confidence = pr.confidence  # 0.98
    plan, notes = taqa_procedure1(pr, eps=eps, p_confidence=p_confidence, db_execute=mock_db_execute, theta_p=0.01)
    print("\nTAQA notes:")
    print("\n".join(notes))

    # 3) rewrite into final SQL using the returned plan
    final_sql, rnotes = rewrite_query_with_plan(pr, plan)
    print("\nRewriter notes:")
    print("\n".join(rnotes))
    print("\nFinal rewritten SQL:\n")
    print(final_sql)

if __name__ == "__main__":
    main()
