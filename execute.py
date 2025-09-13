from typing import Dict, Any
from parser import AQE_Parser
from db_connect import PSQL
import online_query_handler
# import offline_query_handler
# import synopses
from rich.console import Console
from rich.table import Table
import plotext as plt

console = Console()
parser = AQE_Parser()

def plot_query(rows, x_col: str = None, y_col: str = None):
    """
    Plots the result of a SQL query using plotext.
    x_col and y_col should be column names from the query result.
    """
    headers = [desc[0] for desc in rows.description]
    data = rows.fetchall()

    if not data:
        console.print("[red]No data to plot[/red]")
        return

    if x_col is None or y_col is None:
        # default: use first two columns
        x_col, y_col = headers[:2]

    x_idx = headers.index(x_col)
    y_idx = headers.index(y_col)

    x = [row[x_idx] for row in data]
    y = [row[y_idx] for row in data]

    plt.clear_data()
    plt.plot(x, y, marker="dot")
    plt.title(f"{y_col} vs {x_col}")
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.show()


def print_table(rows):
    headers = [desc[0] for desc in rows.description]
    table = Table(show_header=True, header_style="bold magenta")
    for h in headers:
        table.add_column(h, style="dim", overflow="fold")

    for row in rows.fetchall():
        table.add_row(*[str(r) for r in row])

    console.print(table)

def execute_query(raw_sql: str, db: PSQL, user_id: str = None) -> Dict[str, Any]:
    parse_result = parser.parse(raw_sql)
    if parse_result.where and len(parse_result.tables) == 1:
      t = parse_result.tables[0]
      if not getattr(t, "where", None):
          t.where = parse_result.where

    # rows = db.execute(parse_result.cleaned_sql)
    # # print_table(rows)

    # plot_query(rows)
    
    # return {'mode': 'exact', 'result': rows, 'meta': {'note': 'Exact execution (no APPROX)'}}


    # return metadata

    if parse_result.plan_mode == 'exact':
        rows = db.execute(parse_result.cleaned_sql)
        print_table(rows)
        return {'mode': 'exact', 'result': rows, 'meta': {'note': 'Exact execution (no APPROX)'}}

    # if parse_result.plan_mode == 'offline':
    #     approx_result = offline_query_handler.handle_offline(parse_result)
    #     if approx_result.get('success'):
    #         approx_result['mode'] = 'offline'
    #         return approx_result

    if parse_result.plan_mode == 'online':
        try:
            parser.enrich_with_table_stats(parse_result, db)
        except Exception as e:
            # record and continue (TAQA will fallback if it still can't find metadata)
            parse_result.notes.append(f"enrich_with_table_stats error: {e}")
        approx_result = online_query_handler.handle_online(parse_result, db)
        if approx_result.get('success'):
            approx_result['mode'] = 'online'
            return approx_result
        rows = db.execute(parse_result.cleaned_sql)
        print_table(rows)
        return {'plan_mode': 'exact', 'result': rows, 'meta': {'note': 'Fallback exact (approx failed)'}}

    rows = db.execute(parse_result.cleaned_sql)
    return {'plan_mode': 'exact', 'result': rows, 'meta': {'note': 'Default exact execution'}}
