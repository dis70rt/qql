from typing import Dict, Any
from parser import AQE_Parser
from qql import QQLShell
# import online_query_handler
# import offline_query_handler
# import synopses

parser = AQE_Parser()
qql_shell = QQLShell()


def execute_query(raw_sql: str, user_id: str = None) -> Dict[str, Any]:
    parse_result = parser.parse(raw_sql)

    rows = qql_shell.db.execute(parse_result.cleaned_sql)
    return {'mode': 'exact', 'result': rows, 'meta': {'note': 'Exact execution (no APPROX)'}}

    if parse_result.plan_mode == 'exact':
        rows = qql_shell.db.execute(parse_result.cleaned_sql)
        return {'mode': 'exact', 'result': rows, 'meta': {'note': 'Exact execution (no APPROX)'}}

    if parse_result.plan_mode == 'offline':
        approx_result = offline_query_handler.handle_offline(parse_result)
        if approx_result.get('success'):
            approx_result['mode'] = 'offline'
            return approx_result

    if parse_result.plan_mode in ('online', 'offline'):
        approx_result = online_query_handler.handle_online(parse_result)
        if approx_result.get('success'):
            approx_result['mode'] = 'online'
            return approx_result
        rows = qql_shell.db.execute(parse_result.cleaned_sql)
        return {'plan_mode': 'exact', 'result': rows, 'meta': {'note': 'Fallback exact (approx failed)'}}

    rows = qql_shell.db.execute(parse_result.cleaned_sql)
    return {'plan_mode': 'exact', 'result': rows, 'meta': {'note': 'Default exact execution'}}
