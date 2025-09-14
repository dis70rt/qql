import streamlit as st
import timeit
import re
from decimal import Decimal
from typing import Any, Dict, List, Tuple, Optional
import plotly.graph_objects as go

from utils.db_connect import PSQL
from utils.db_connect import QQLEnv
from execute import execute_query

def _ensure_semicolon(sql: str) -> str:
    s = sql.strip()
    return s if s.endswith(";") else s + ";"


def _strip_approx_tokens(sql: str) -> str:
    s = re.sub(r"\bAPPROX\b", "", sql, flags=re.IGNORECASE)
    s = re.sub(r"\bERROR\s+[0-9]*\.?[0-9]+\b", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\bPROB(?:ABILITY)?\s+[0-9]*\.?[0-9]+\b", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\bCONFIDENCE\s+[0-9]*\.?[0-9]+\b", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+", " ", s).strip()
    return _ensure_semicolon(s)


def _inject_approx(sql: str, eps: float, conf: float) -> str:
    s = _strip_approx_tokens(sql)
    s = re.sub(r"^(\s*SELECT\s+)", r"\1APPROX ", s, flags=re.IGNORECASE)
    s = s[:-1] + f" ERROR {eps} CONFIDENCE {conf};"
    return s


def _to_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, Decimal):
        try:
            return float(x)
        except Exception:
            return None
    try:
        return float(x)
    except Exception:
        return None


def _rows_to_df(rows: List[Tuple], cols: List[str]):
    try:
        import pandas as pd
        return pd.DataFrame(rows, columns=cols)
    except Exception:
        return None


st.set_page_config(page_title="QQL: Approx vs Exact", layout="wide")
st.title("QQL: Compare APPROX vs EXACT")

with st.sidebar:
    st.header("Database Login")
    env = QQLEnv()
    default_host = env.host or "localhost"
    default_port = int(env.port or 5432)
    default_db = env.dbname or "postgres"
    default_user = env.username or "postgres"
    default_pass = env.password or ""

    with st.form("db_login"):
        host = st.text_input("Host", value=default_host)
        port = st.number_input("Port", min_value=1, max_value=65535, value=default_port, step=1)
        dbname = st.text_input("Database", value=default_db)
        username = st.text_input("Username", value=default_user)
        password = st.text_input("Password", value=default_pass, type="password")
        submitted = st.form_submit_button("Connect")

    if submitted:
        try:
            db = PSQL(dbname=dbname, user=username, password=password, host=host, port=str(port))
            st.session_state["db"] = db
            st.success("Connected to database")
        except Exception as e:
            st.error(f"Connection failed: {e}")

DB: Optional[PSQL] = st.session_state.get("db")

st.markdown("---")

st.subheader("Enter SQL query (without APPROX)")
def_q = "SELECT AVG(price) FROM pizza_orders;"
user_sql = st.text_area("SQL", value=def_q, height=120, help="Provide a standard SQL query. APPROX/ERROR/CONFIDENCE will be added automatically.")

col_a, col_b = st.columns(2)
with col_a:
    eps = st.slider("ERROR (epsilon)", min_value=0.01, max_value=0.60, value=0.10, step=0.01)
with col_b:
    conf = st.slider("CONFIDENCE", min_value=0.50, max_value=0.999, value=0.95, step=0.01)

run_btn = st.button("Run Comparison")

if run_btn:
    if DB is None:
        st.error("Please connect to the database first from the sidebar.")
    else:
        exact_sql = _strip_approx_tokens(user_sql)
        approx_sql = _inject_approx(user_sql, eps=eps, conf=conf)

        st.code(exact_sql, language="sql")
        st.code(approx_sql, language="sql")

        t0 = timeit.default_timer()
        approx_result: Dict[str, Any] = execute_query(approx_sql, DB)
        t1 = timeit.default_timer()
        approx_time = t1 - t0

        t0 = timeit.default_timer()
        cur = DB.execute(exact_sql)
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description] if cur.description else []
        exact_result: Dict[str, Any] = {"rows": rows, "columns": cols}
        t1 = timeit.default_timer()
        exact_time = t1 - t0

        st.success("Queries executed")

        speedup = (exact_time / approx_time) if approx_time > 0 else 0.0
        m1, m2, m3 = st.columns(3)
        m1.metric("Approx Time", f"{approx_time:.4f} s")
        m2.metric("Exact Time", f"{exact_time:.4f} s")
        m3.metric("Speedup (Exact/Approx)", f"{speedup:.2f}x")

        fig = go.Figure()
        fig.add_bar(x=["Approx", "Exact"], y=[approx_time, exact_time], marker_color=["#1f77b4", "#ff7f0e"])
        fig.update_layout(title="Runtime Comparison", yaxis_title="Time (s)", template="plotly_white", bargap=0.35)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.subheader("Results")
        c1, c2 = st.columns(2)

        with c1:
            st.markdown("**Approximate Result**")
            estimate_block = approx_result.get("estimate") if isinstance(approx_result, dict) else None
            approx_value = None
            approx_ci = (None, None)
            if isinstance(estimate_block, dict):
                est = estimate_block.get("estimate")
                ci_lower = estimate_block.get("ci_lower")
                ci_upper = estimate_block.get("ci_upper")
                cols = estimate_block.get("columns") or approx_result.get("columns")
                if isinstance(est, list) and cols:
                    df = _rows_to_df(est, cols)
                    if df is not None:
                        st.dataframe(df)
                    else:
                        st.write(est)
                else:
                    approx_value = _to_float(est)
                    if approx_value is not None:
                        st.metric("Estimate", f"{approx_value:.6f}")
                    else:
                        st.write(est)
                if ci_lower is not None and ci_upper is not None and not isinstance(ci_lower, list):
                    approx_ci = (_to_float(ci_lower), _to_float(ci_upper))
                    st.caption(f"CI: [{approx_ci[0]}, {approx_ci[1]}]")
            else:
                st.write(approx_result)

        with c2:
            st.markdown("**Exact Result**")
            rows = exact_result.get("rows", [])
            cols = exact_result.get("columns", [])
            if rows and cols:
                if len(rows) == 1 and len(cols) == 1:
                    exact_value = _to_float(rows[0][0])
                    if exact_value is not None:
                        st.metric("Exact", f"{exact_value:.6f}")
                    else:
                        st.write(rows)
                else:
                    df = _rows_to_df(rows, cols)
                    if df is not None:
                        st.dataframe(df)
                    else:
                        st.write(rows)
            else:
                st.write(exact_result)

        try:
            exact_scalar = _to_float(exact_result["rows"][0][0]) if exact_result.get("rows") and len(exact_result["rows"]) == 1 and len(exact_result.get("columns", [])) == 1 else None
        except Exception:
            exact_scalar = None
        approx_scalar = approx_value
        if exact_scalar is not None and approx_scalar is not None:
            abs_err = abs(approx_scalar - exact_scalar)
            rel_err = abs_err / (abs(exact_scalar) if exact_scalar != 0 else 1.0)
            e1, e2 = st.columns(2)
            e1.metric("Absolute Error", f"{abs_err:.6f}")
            e2.metric("Relative Error", f"{rel_err:.4%}")

st.caption("Note: LIMIT/OFFSET may be ignored for approximate planning; DISTINCT and window functions fall back to exact execution.")
