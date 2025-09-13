from typing import Callable, Dict, List, Tuple, Optional
import math

from handler.online.query_sampler import SamplePlan, SamplePlanEntry, build_sample_subquery
import utils.bsap as bsap


def _compute_block_stats_from_sums(block_sums):
    n = len(block_sums)
    if n == 0:
        return 0.0, 0.0, 0
    mean = sum(block_sums) / n
    if n <= 1:
        var = 0.0
    else:
        var = sum((x - mean) ** 2 for x in block_sums) / (n - 1)
    return mean, var, n

def taqa_procedure1(parse_result,
                    eps: float,
                    p_confidence: float,
                    db_execute: Callable[[str], Dict],
                    theta_p: float = 0.005,
                    delta1: Optional[float] = None,
                    delta2: Optional[float] = None,
                    max_pilot_blocks: int = 500) -> Tuple[SamplePlan, List[str]]:

    notes: List[str] = []
    plan = SamplePlan()

    if delta1 is None or delta2 is None:
        spare = max(0.0, 1.0 - p_confidence)
        delta1 = delta1 if delta1 is not None else spare / 3.0
        delta2 = delta2 if delta2 is not None else spare / 3.0

    p_prime = p_confidence + delta1 + delta2
    if p_prime >= 1.0:
        p_prime = 0.999
        notes.append("p' clipped to 0.999 due to large deltas")

    if not parse_result.aggregates:
        notes.append("No aggregates found; empty plan returned.")
        return plan, notes

    for agg in parse_result.aggregates:
        # choose table
        if len(parse_result.tables) == 0:
            notes.append("No table metadata; cannot run TAQA Procedure 1 for this aggregate")
            continue
        if len(parse_result.tables) == 1:
            tbl = parse_result.tables[0]
        else:
            tbl = parse_result.tables[0]
            notes.append("Multiple tables present: using first table heuristically for Procedure1 (join-aware TAQA not implemented).")

        alias = tbl.alias or tbl.name

        B = getattr(tbl, "num_blocks", None)
        if not B:
            notes.append(f"Table {alias} has no num_blocks metadata; cannot run block-TAQA. Marking as full (no sampling).")
            entry = SamplePlanEntry(table=tbl, method="full", sample_frac=None)
            plan.entries[alias] = entry
            continue

        pilot_blocks = max(2, int(math.ceil(B * theta_p)))
        pilot_frac_blocks = pilot_blocks / float(B)

        pilot_sql = f"-- PILOT BLOCK SAMPLE: table={tbl.name} pilot_frac_blocks={pilot_frac_blocks}\nSELECT /* pilot */ FROM {tbl.name} TABLESAMPLE SYSTEM ({pilot_frac_blocks * 100.0});"
        res = db_execute(pilot_sql)
        stats = res.get("sample_stats", {})

        key_alias = getattr(agg, "alias", None)
        key_candidates = []
        if key_alias:
            key_candidates.append(key_alias)
        key_candidates.append(f"{agg.func}({agg.column})")
        key_candidates.append(f"{agg.func.lower()}({agg.column})")

        s = None
        for k in key_candidates:
            if k in stats:
                s = stats[k]
                break

        if not s:
            notes.append(f"No pilot stats returned for aggregate {agg.func}({agg.column}) on table {alias}; leaving as full table")
            entry = SamplePlanEntry(table=tbl, method="full", sample_frac=None)
            plan.entries[alias] = entry
            continue

        func_lower = agg.func.lower()
        if func_lower == "avg":
            a_blocks = s.get("block_sums")
            b_blocks = s.get("block_counts")

            if a_blocks is None and "block_pairs" in s:
                pairs = s["block_pairs"]
                a_blocks = [p[0] for p in pairs]
                b_blocks = [p[1] for p in pairs]

            if a_blocks is None or b_blocks is None:
                notes.append(f"Pilot stats incomplete for AVG on {alias}; expected block_sums & block_counts; got keys {list(s.keys())}. Falling back to full.")
                entry = SamplePlanEntry(table=tbl, method="full", sample_frac=None)
                plan.entries[alias] = entry
                continue

            a_blocks = [float(x) for x in a_blocks]
            b_blocks = [float(x) for x in b_blocks]
            n_p = len(a_blocks)
            if n_p == 0:
                notes.append(f"Pilot returned zero blocks for {alias}; falling back to full.")
                entry = SamplePlanEntry(table=tbl, method="full", sample_frac=None)
                plan.entries[alias] = entry
                continue
            mean_a = float(sum(a_blocks) / n_p)
            mean_b = float(sum(b_blocks) / n_p)
            # unbiased sample variances
            if n_p > 1:
                var_a = sum((x - mean_a) ** 2 for x in a_blocks) / (n_p - 1)
                var_b = sum((x - mean_b) ** 2 for x in b_blocks) / (n_p - 1)
                cov_ab = sum((x - mean_a) * (y - mean_b) for x, y in zip(a_blocks, b_blocks)) / (n_p - 1)
            else:
                var_a = var_b = cov_ab = 0.0

            muA_hat_p = float(B) * float(mean_a)
            muB_hat_p = float(B) * float(mean_b)

            var_muA_hat_p = (float(B) ** 2) * (1.0 - (n_p / float(B))) * (float(var_a) / float(max(1, n_p)))
            var_muB_hat_p = (float(B) ** 2) * (1.0 - (n_p / float(B))) * (float(var_b) / float(max(1, n_p)))
            cov_muA_muB_hat_p = (float(B) ** 2) * (1.0 - (n_p / float(B))) * (float(cov_ab) / float(max(1, n_p)))

            theta_hat_p = muA_hat_p / max(1e-18, muB_hat_p)

            var_theta_p = (1.0 / (muB_hat_p ** 2)) * var_muA_hat_p \
                          - (2.0 * muA_hat_p / (muB_hat_p ** 3)) * cov_muA_muB_hat_p \
                          + (muA_hat_p ** 2 / (muB_hat_p ** 4)) * var_muB_hat_p

            # Lower bound for theta (L_theta)
            L_theta = bsap.L_mu_lower(theta_hat_p, var_theta_p, delta1)

            if L_theta <= 0:
                notes.append(f"L_theta <= 0 for table {alias} (L_theta={L_theta}); Procedure1 may not be usable; consider absolute error or larger pilot_frac.")

            quantile = (1.0 + p_prime) / 2.0
            z_pp = bsap.z_from_quantile(quantile)

            chosen_n = None
            chosen_uv = None

            for n in range(1, B + 1):
                uv_muA = (float(B) ** 2) * (1.0 - (n / float(B))) * (float(var_a) / float(max(1, n)))
                uv_muB = (float(B) ** 2) * (1.0 - (n / float(B))) * (float(var_b) / float(max(1, n)))
                uv_cov = (float(B) ** 2) * (1.0 - (n / float(B))) * (float(cov_ab) / float(max(1, n)))

                uv_theta = (1.0 / (muB_hat_p ** 2)) * uv_muA \
                           - (2.0 * muA_hat_p / (muB_hat_p ** 3)) * uv_cov \
                           + (muA_hat_p ** 2 / (muB_hat_p ** 4)) * uv_muB

                uv_theta = max(0.0, uv_theta)

                lhs = z_pp * math.sqrt(max(0.0, uv_theta)) / max(1e-18, L_theta)
                if lhs <= eps:
                    chosen_n = n
                    chosen_uv = uv_theta
                    break

            if chosen_n is None:
                notes.append(f"No n <= B meets Eq(6) for AVG {alias}; returning full table as fallback.")
                entry = SamplePlanEntry(table=tbl, method="full", sample_frac=None)
                plan.entries[alias] = entry
                continue

            frac = chosen_n / float(B)
            subq_sql, weight_expr = build_sample_subquery(tbl, frac, method="block")
            entry = SamplePlanEntry(table=tbl, method="block", blocks_to_sample=chosen_n, subquery_sql=subq_sql, weight_expr=weight_expr)
            entry.mu_hat_p = theta_hat_p
            entry.var_mu_hat_p = var_theta_p
            entry.chosen_uv = chosen_uv
            entry.L_mu = L_theta
            entry.pilot_n_blocks = n_p
            entry.num_blocks = B
            plan.entries[alias] = entry
            notes.append(f"Procedure1 table={alias}: pilot_blocks={n_p}, theta_hat_p={theta_hat_p:.6f}, L_theta={L_theta:.6f}, chosen_n={chosen_n}/{B}, frac={frac:.6f}")
            continue

        if "block_sums" in s:
            block_mean, block_var, n_p = _compute_block_stats_from_sums(s["block_sums"])
        else:
            block_mean = s.get("block_mean")
            block_var = s.get("block_var")
            n_p = s.get("n_blocks") or pilot_blocks

        mu_hat_p = float(B) * float(block_mean)
        var_mu_hat_p = (float(B) ** 2) * (1.0 - (n_p / float(B))) * (float(block_var) / float(max(1, n_p)))
        L_mu = bsap.L_mu_lower(mu_hat_p, var_mu_hat_p, delta1)
        if L_mu <= 0:
            notes.append(f"L_mu <= 0 for table {alias} (L_mu={L_mu}); Procedure1 may not be usable; consider using absolute error or larger pilot_frac.")

        quantile = (1.0 + p_prime) / 2.0
        z_pp = bsap.z_from_quantile(quantile)

        chosen_n = None
        chosen_uv = None
        for n in range(1, B + 1):
            uv = bsap.blocks_uv(B, n, block_var)
            lhs = z_pp * math.sqrt(max(0.0, uv)) / max(1e-18, L_mu)
            if lhs <= eps:
                chosen_n = n
                chosen_uv = uv
                break

        if chosen_n is None:
            notes.append(f"No n <= B meets Eq(6) for {alias}; returning full table as fallback.")
            entry = SamplePlanEntry(table=tbl, method="full", sample_frac=None)
            plan.entries[alias] = entry
            continue

        frac = chosen_n / float(B)
        subq_sql, weight_expr = build_sample_subquery(tbl, frac, method="block")
        entry = SamplePlanEntry(table=tbl, method="block", blocks_to_sample=chosen_n, subquery_sql=subq_sql, weight_expr=weight_expr)
        entry.mu_hat_p = mu_hat_p
        entry.var_mu_hat_p = var_mu_hat_p
        entry.chosen_uv = chosen_uv
        entry.L_mu = L_mu
        entry.pilot_n_blocks = n_p
        entry.num_blocks = B
        plan.entries[alias] = entry
        notes.append(f"Procedure1 table={alias}: pilot_blocks={n_p}, mu_hat_p={mu_hat_p:.6f}, L_mu={L_mu:.6f}, chosen_n={chosen_n}/{B}, frac={frac:.6f}")

    return plan, notes