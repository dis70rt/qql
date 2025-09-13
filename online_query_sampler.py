# sampler.py
from dataclasses import dataclass, field
from typing import Optional, Dict
import re
from parser import TableInfo

@dataclass
class SamplePlanEntry:
    table: TableInfo
    method: str                    # 'block' | 'row_system' | 'row_rand' | 'full'
    sample_frac: Optional[float] = None
    blocks_to_sample: Optional[int] = None
    subquery_sql: Optional[str] = None
    weight_expr: Optional[str] = None
    mu_hat_p: Optional[float] = None   # mu estimate
    var_mu_hat_p: Optional[float] = None
    chosen_uv: Optional[float] = None  # U_V[Theta] for chosen plan
    L_mu: Optional[float] = None
    pilot_n_blocks: Optional[int] = None
    num_blocks: Optional[int] = None


@dataclass
class SamplePlan:
    entries: Dict[str, SamplePlanEntry] = field(default_factory=dict)
    notes: list = field(default_factory=list)


def _qualify_alias(table: TableInfo) -> str:
    return table.alias or re.sub(r'[^0-9A-Za-z_]', '_', table.name)


def build_sample_subquery(table: TableInfo, sample_frac: float, method: str = 'block', weight_col: str = '_sample_weight'):
    """
    - method == 'block' : TABLESAMPLE SYSTEM(percent) assumed to select disk blocks/pages.
    - weight_expr is the Horvitz-Thompson per-row weight (1 / inclusion_prob). For block sampling,
      if frac = n / B then weight = (1.0 / frac) = B / n.
    Returns (subquery_sql, weight_expr).
    """
    if not (0.0 < sample_frac <= 1.0):
        raise ValueError("sample_frac must be in (0,1].")

    alias = _qualify_alias(table)
    sampled_alias = f"{alias}__s"
    weight_literal = f"(1.0/{sample_frac})"
    where_clause = f" WHERE {table.where}" if getattr(table, "where", None) else ""
    percent = sample_frac * 100.0

    if method in ("block", "row_system"):
        subquery = (
            f"(SELECT t.*, {weight_literal} AS {weight_col} "
            f"FROM {table.name} AS t TABLESAMPLE SYSTEM ({percent}){where_clause}) AS {sampled_alias}"
        )
        return subquery, weight_literal
    elif method == "row_rand":
        where_pred = f"random() < {sample_frac}"
        where_clause = f" WHERE {table.where} AND {where_pred}" if getattr(table, "where", None) else f" WHERE {where_pred}"
        subquery = f"(SELECT t.*, {weight_literal} AS {weight_col} FROM {table.name} AS t{where_clause}) AS {sampled_alias}"
        return subquery, weight_literal
    elif method == "full":
        qualified = f"{table.name} AS {alias}"
        return qualified, "1.0"
    else:
        raise ValueError(f"unknown sampling method '{method}'")
