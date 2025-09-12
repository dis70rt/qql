import sqlglot
from sqlglot import exp
from sqlglot.dialects.dialect import Dialect
from sqlglot.parser import Parser as SqlglotParser
from sqlglot.generator import Generator
from sqlglot.tokens import Tokenizer, TokenType

class Error(exp.Expression):
    arg_types = {"this": True, "percent": False}

class QQL(Dialect):
    class Tokenizer(Tokenizer):
        KEYWORDS = {
            **Tokenizer.KEYWORDS,
            "ERROR": TokenType.VAR,
            "PERCENT": TokenType.VAR,
        }

    class Parser(SqlglotParser):
        def _parse_select(self, *args, **kwargs):
            select = super()._parse_select(*args, **kwargs)
            
            if self._match_text_seq("ERROR"):
                self._retreat(self._index - 1)
                error_clause = self._parse_error_clause()
                if error_clause:
                    select.args["error"] = error_clause
            
            return select

        def _parse_error_clause(self):
            if self._match_text_seq("ERROR"):
                value = self._parse_primary()
                if not value:
                    self.error("Expected value after ERROR")
                
                percent = self._match_text_seq("PERCENT")
                return self.expression(Error, this=value, percent=bool(percent))
            return None

    class Generator(Generator):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.TYPE_MAPPING[Error] = self.error_sql
        
        def error_sql(self, expression: Error) -> str:
            result = f"ERROR {self.sql(expression, 'this')}"
            if expression.args.get('percent'):
                result += " PERCENT"
            return result

        def select_sql(self, expression: exp.Select) -> str:
            sql = super().select_sql(expression)
            error_clause = expression.args.get('error')
            if error_clause:
                return f"{sql} {self.sql(error_clause)}"
            return sql

if __name__ == '__main__':
    try:
        sql_query = "SELECT column_a FROM my_table WHERE x > 1 ERROR 5 PERCENT"
        print(f"Parsing: {sql_query}")
        parsed_ast = sqlglot.parse_one(sql_query, read=QQL)
        
        error_node = parsed_ast.args.get("error")
        if error_node:
            value = error_node.this
            has_percent = error_node.args.get("percent", False)
            
            print(f"Extracted Error Value: {value}")
            print(f"Has PERCENT keyword: {has_percent}")
        else:
            print("No ERROR clause found")
            
        print("\nGenerated SQL:")
        print(parsed_ast.sql(dialect=QQL, pretty=True))
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()