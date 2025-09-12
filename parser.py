from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
import re
import sqlglot
from sqlglot import exp

DEFAULT_ERROR = 0.05
DEFAULT_CONFIDENCE = 0.95

KW_APPROX = "APPROX"
KW_ERROR = "ERROR"
KW_PROBABILITY = "PROBABILITY"
KW_PROB = "PROB"
KW_CONFIDENCE = "CONFIDENCE"


@dataclass
class AggregateInfo:
    func: str
    column: Optional[str]
    raw_ast: exp.Expression


@dataclass
class TableInfo:
    name: str
    alias: Optional[str]


@dataclass
class JoinInfo:
    left_table: str
    right_table: str
    condition: Optional[str]


@dataclass
class ParseResult:
    original_sql: str
    cleaned_sql: str
    approx: bool
    error: float
    confidence: float
    plan_mode: str  # 'exact' | 'offline' | 'online'
    ast: Optional[exp.Expression]    # parsed AST of cleaned sql
    aggregates: List[AggregateInfo] = field(default_factory=list)
    tables: List[TableInfo] = field(default_factory=list)
    joins: List[JoinInfo] = field(default_factory=list)
    group_by: List[str] = field(default_factory=list)
    where: Optional[str] = None
    ctes: Dict[str, str] = field(default_factory=dict)  # alias -> SQL string
    contains_distinct: bool = False
    contains_window: bool = False
    contains_subqueries: bool = False
    is_rewritable: bool = True
    notes: List[str] = field(default_factory=list)  # helpful diagnostics


class AQE_Parser:
    
    def __init__(self, default_error: float = DEFAULT_ERROR, default_confidence: float = DEFAULT_CONFIDENCE):
        self.default_error = default_error
        self.default_confidence = default_confidence

    _error_re = re.compile(r'(?i)\bERROR\s*(?:=)?\s*([0-9]*\.?[0-9]+)\b')
    _prob_re = re.compile(r'(?i)\b(?:PROBABILITY|PROB|CONFIDENCE)\s*(?:=)?\s*([0-9]*\.?[0-9]+)\b')
    _approx_after_select_re = re.compile(r'(?i)^\s*SELECT\s+APPROX\b')


    def _extract_keywords(self, sql: str) -> Tuple[str, bool, Optional[float], Optional[float]]:
        approx = False
        error_val = None
        prob_val = None

        if self._approx_after_select_re.search(sql):
            approx = True
            sql = re.sub(r'(?i)^\s*(SELECT)\s+APPROX\b', r'\1', sql, count=1)
        m = self._error_re.search(sql)
        if m:
            try:
                error_val = float(m.group(1))
                sql = self._error_re.sub('', sql, count=1)
            except Exception:
                error_val = None
        m2 = self._prob_re.search(sql)
        if m2:
            try:
                prob_val = float(m2.group(1))
                sql = self._prob_re.sub('', sql, count=1)
            except Exception:
                prob_val = None

        sql = re.sub(r'\s+', ' ', sql).strip()
        return sql, approx, error_val, prob_val


    def _parse_sql(self, sql: str) -> Optional[exp.Expression]:
        try:
            ast = sqlglot.parse_one(sql, read='postgres')
            return ast
        except Exception as e:
            return None


    def _extract_ctes(self, ast: exp.Expression) -> Dict[str, str]:
        """Collect CTEs (WITH ... AS ...) into a dict."""
        ctes = {}
        for node in ast.find_all(exp.CTE):
            try:
                alias = node.alias
                ctes[alias] = node.this.sql()
            except Exception:
                pass
        return ctes

    def _find_tables_and_aliases(self, ast: exp.Expression) -> List[TableInfo]:
        tables = []
        for t in ast.find_all(exp.Table):
            try:
                name = t.this.this
                alias = None
                if t.alias:
                    alias = t.alias
                tables.append(TableInfo(name=name, alias=alias))
            except Exception:
                continue
        return tables

    def _find_aggregates(self, ast: exp.Expression) -> List[AggregateInfo]:
        """Return aggregates (SUM/AVG/COUNT/other AggFunc) found in SELECT list."""
        aggs = []
        for agg in ast.find_all(exp.AggFunc):
            try:
                func_name = type(agg).__name__.upper()
                func_token = agg.name.upper() if hasattr(agg, 'name') and agg.name else agg.sql().split('(')[0].upper()
                # column: get the first column expression inside agg if present
                column_expr = None
                col_node = agg.find(exp.Column)
                if col_node:
                    column_expr = col_node.sql()
                aggs.append(AggregateInfo(func=func_token, column=column_expr, raw_ast=agg))
            except Exception:
                continue
        return aggs

    def _find_group_by(self, ast: exp.Expression) -> List[str]:
        groups = []
        for g in ast.find_all(exp.Group):
            if 'expressions' in g.args:
                for e in g.args['expressions']:
                    groups.append(e.sql())
            elif 'rollup' in g.args:
                for e in g.args['rollup']:
                    groups.append(e.sql())
        return groups

    def _find_where(self, ast: exp.Expression) -> Optional[str]:
        where = ast.find(exp.Where)
        return where.this.sql() if where else None

    def _find_joins(self, ast: exp.Expression) -> List[JoinInfo]:
        joins = []
        for j in ast.find_all(exp.Join):
            try:
                left_table = None
                right_table = None
                tables = j.find_all(exp.Table)
                tables = list(tables)
                if len(tables) >= 2:
                    left_table = tables[0].this.this
                    right_table = tables[1].this.this
                elif len(tables) == 1:
                    right_table = tables[0].this.this
                on = j.args.get('on')
                cond = on.sql() if on is not None else None
                joins.append(JoinInfo(left_table=left_table, right_table=right_table, condition=cond))
            except Exception:
                continue
        return joins

    def _detect_distinct(self, ast: exp.Expression) -> bool:
        for sel in ast.find_all(exp.Select):
            if sel.args.get('distinct'):
                return True
        return False

    def _detect_window_functions(self, ast: exp.Expression) -> bool:
        for node in ast.find_all(exp.Window):
            return True
        for func in ast.find_all(exp.Func):
            if func.args.get('over'):
                return True
        return False

    def _detect_subqueries(self, ast: exp.Expression) -> bool:
        return any(isinstance(n, exp.Select) and n is not ast for n in ast.find_all(exp.Select))

    def _detect_unsupported_features(self, ast: exp.Expression) -> Tuple[bool, List[str]]:
        notes = []
        unsupported = False

        if self._detect_window_functions(ast):
            unsupported = True
            notes.append("Query contains window functions (OVER) — not supported for approximate rewrite.")

        if self._detect_distinct(ast):
            unsupported = True
            notes.append("DISTINCT detected — prefer sketches (HLL / t-digest). Not supported for page-level BSAP rewrite.")

        if ast.find(exp.Order):
            notes.append("ORDER BY present. Sampling semantics may not match expected ordering. Consider removing ORDER BY for approximate execution.")

        if ast.find(exp.Limit) or ast.find(exp.Offset):
            notes.append("LIMIT/OFFSET detected — these are removed for pilot rewrite (results may differ).")

        if ast.find(exp.Having):
            notes.append("HAVING clause detected — may complicate sample-based guarantees; handled conservatively.")

        if ast.find(exp.SetOperation):
            unsupported = True
            notes.append("Set operations (UNION/INTERSECT/EXCEPT) detected — only simple UNION is supported in limited ways.")

        return unsupported, notes


    def parse(self, original_sql: str) -> ParseResult:
        cleaned_sql, approx_flag, err_val, prob_val = self._extract_keywords(original_sql)
        error = err_val if err_val is not None else self.default_error
        confidence = prob_val if prob_val is not None else self.default_confidence

        if not approx_flag:
            plan_mode = 'exact'
        else:
            if err_val is not None or prob_val is not None:
                plan_mode = 'online'
            else:
                plan_mode = 'offline'

        ast = self._parse_sql(cleaned_sql)
        aggregates = []
        tables = []
        joins = []
        group_by = []
        where = None
        ctes = {}
        contains_distinct = False
        contains_window = False
        contains_subqueries = False
        is_rewritable = True
        notes = []

        if ast is None:
            notes.append("Failed to parse SQL with sqlglot - non-rewritable.")
            parse_result = ParseResult(
                original_sql=original_sql,
                cleaned_sql=cleaned_sql,
                approx=approx_flag,
                error=error,
                confidence=confidence,
                plan_mode='exact' if not approx_flag else 'online',  # be conservative
                ast=None,
                is_rewritable=False,
                notes=notes
            )
            return parse_result

        ctes = self._extract_ctes(ast)
        tables = self._find_tables_and_aliases(ast)
        aggregates = self._find_aggregates(ast)
        group_by = self._find_group_by(ast)
        where = self._find_where(ast)
        joins = self._find_joins(ast)
        contains_distinct = self._detect_distinct(ast)
        contains_window = self._detect_window_functions(ast)
        contains_subqueries = self._detect_subqueries(ast)

        unsupported, unsupported_notes = self._detect_unsupported_features(ast)
        if unsupported:
            is_rewritable = False
            notes.extend(unsupported_notes)

        for agg in aggregates:
            # if agg.raw_ast has multiple column tables inside → join-aware
            cols = set()
            for col in agg.raw_ast.find_all(exp.Column):
                if col.table:
                    cols.add(col.table)
            if len(cols) > 1:
                notes.append(f"Aggregate {agg.func} references columns from multiple tables: {cols} -> requires join-aware sampling.")
                # not necessarily non-rewritable, but planner must know to take care
                # keep is_rewritable True but add note

        parse_result = ParseResult(
            original_sql=original_sql,
            cleaned_sql=cleaned_sql,
            approx=approx_flag,
            error=error,
            confidence=confidence,
            plan_mode=plan_mode,
            ast=ast,
            aggregates=aggregates,
            tables=tables,
            joins=joins,
            group_by=group_by,
            where=where,
            ctes=ctes,
            contains_distinct=contains_distinct,
            contains_window=contains_window,
            contains_subqueries=contains_subqueries,
            is_rewritable=is_rewritable,
            notes=notes
        )
        return parse_result


    def page_id_expr(self, label_index: int = 0) -> exp.Expression:
        """
        Return a sqlglot AST expression for a page_id using Postgres ctid, e.g.:

            'page_id_0:' || (table.ctid::text::point)[0]::int as page_id_0

        NOTE: substitute the correct table identifier into the template
        or use the AST to place it in a SELECT. This helper returns a *template*
        AST using a placeholder name '<<<LARGEST_TABLE>>>' that should be
        replaced with the actual table alias/name before rendering SQL.
        """
        tbl = "<<<LARGEST_TABLE>>>"
        page_id_sql = f"'page_id_{label_index}:' || ({tbl}.ctid::text::point)[0]::int as page_id_{label_index}"
        return sqlglot.parse_one(page_id_sql, read='postgres')

# # for testing
# if __name__ == "__main__":
#     demo_queries = [
#         "SELECT APPROX SUM(amount) FROM orders o JOIN customers c ON o.customer_id=c.id WHERE c.region='APAC' ERROR 0.03 PROB 0.98",
#         "SELECT APPROX SUM(amount) FROM orders WHERE created_at > now() - interval '7' days",
#         "SELECT SUM(amount) FROM orders WHERE created_at > now() - interval '7' days",
#         # "SELECT APPROX COUNT(DISTINCT user_id) FROM events",  # should warn about distinct
#     ]

#     parser = AQE_Parser()
#     for q in demo_queries:
#         print("\n---\nOriginal:", q)
#         r = parser.parse(q)
#         print("Cleaned SQL:", r.cleaned_sql)
#         print("Approx enabled:", r.approx, "Plan mode:", r.plan_mode)
#         print("Error / Confidence:", r.error, "/", r.confidence)
#         print("Tables:", [(t.name, t.alias) for t in r.tables])
#         print("Aggregates:", [(a.func, a.column) for a in r.aggregates])
#         print("Group by:", r.group_by)
#         print("Where:", r.where)
#         print("Rewritable:", r.is_rewritable)
#         print("AST:", repr(r.ast))
#         if r.notes:
#             print("Notes:", r.notes)
