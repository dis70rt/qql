from typing import Dict, Any
from parser import AQE_Parser
from utils.db_connect import PSQL
from handler.online import query_handler as oqh
# import offline_query_handler
# import synopses
from rich.console import Console
from rich.table import Table
from decimal import Decimal
import re

console = Console()
parser = AQE_Parser()

def print_table(rows):
    headers = [desc[0] for desc in rows.description]
    table = Table(show_header=True, header_style="bold magenta")
    for h in headers:
        table.add_column(h, style="dim", overflow="fold")

    for row in rows.fetchall():
        table.add_row(*[str(r) for r in row])

    console.print(table)


def _extract_header_from_sql(sql_query: str) -> str:
    match = re.search(r"(\w+)\s*\(\s*\(?([\w_]+)", sql_query, re.IGNORECASE)
    
    if match:
        agg_function = match.group(1).upper()
        column_name = match.group(2).replace("__s", "")
        return f"{agg_function}({column_name})"
        
    return "aggregate_result"

def print_dict(data_dict: dict):
    console = Console()
    estimate_details = data_dict.get('estimate', {})
    value = estimate_details.get('estimate')
    columns = estimate_details.get('columns') or data_dict.get('columns')
    if isinstance(value, list) and value and isinstance(value[0], (list, tuple)):
        table = Table(show_header=True, header_style="bold magenta")
        if columns:
            for h in columns:
                table.add_column(str(h), style="dim", overflow="fold")
        else:
            ncols = len(value[0])
            for i in range(ncols):
                table.add_column(f"col_{i}", style="dim", overflow="fold")
        for row in value:
            table.add_row(*[str(x) for x in row])
        console.print(table)
        return
    table = Table(show_header=True, header_style="bold magenta")
    sql_query = data_dict.get('final_sql', '')
    header = _extract_header_from_sql(sql_query)
    table.add_column(header, style="dim", overflow="fold")
    row_to_add = "N/A"
    if value is not None:
        if isinstance(value, Decimal):
            row_to_add = f"{value:.2f}"
        else:
            row_to_add = str(value)
    table.add_row(row_to_add)
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
        approx_result = oqh.handle_online(parse_result, db)
        if approx_result.get('success'):
            approx_result['mode'] = 'online'
            print_dict(approx_result)
            return approx_result
        
        rows = db.execute(parse_result.cleaned_sql)
        return {'plan_mode': 'exact', 'result': rows, 'meta': {'note': 'Fallback exact (approx failed)'}}

    rows = db.execute(parse_result.cleaned_sql)
    print_table(rows)
    return {'plan_mode': 'exact', 'result': rows, 'meta': {'note': 'Default exact execution'}}
