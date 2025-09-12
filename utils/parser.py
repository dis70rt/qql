from sqlglot import exp
from sqlglot.dialects.dialect import Dialect
from sqlglot.parser import Parser as SqlglotParser
from sqlglot.generator import Generator
from sqlglot.tokens import Tokenizer, TokenType

TokenType.APPROX = "APPROX"
TokenType.ERROR = "ERROR"
TokenType.CONFIDENCE = "CONFIDENCE"

class Error(exp.Expression):
    arg_types = {"this": True}

class Confidence(exp.Expression):
    arg_types = {"this": True}

class Approx(exp.Expression):
    arg_types = {"this": True}

class QQL(Dialect):
    class Tokenizer(Tokenizer):
        KEYWORDS = {
            **Tokenizer.KEYWORDS,
            "ERROR": TokenType.ERROR,
            "CONFIDENCE": TokenType.CONFIDENCE,
            "APPROX": TokenType.APPROX,
        }

        SINGLE_TOKENS = {
            **Tokenizer.SINGLE_TOKENS,
            "ERROR": TokenType.ERROR,
            "CONFIDENCE": TokenType.CONFIDENCE,
            "APPROX": TokenType.APPROX,
        }

    class Parser(SqlglotParser):
        def _parse_select_query(self, *args, **kwargs):
            approx = self._match_text_seq("APPROX")
            select = super()._parse_select_query(*args, **kwargs)

            if approx:
                select.args["approx"] = True
            
            if self._match(TokenType.ERROR):
                value = self._parse_primary()
                if value:
                    select.args["error"] = self.expression(Error, this=value)
            
            if self._match(TokenType.CONFIDENCE):
                value = self._parse_primary()
                if value:
                    select.args["confidence"] = self.expression(Confidence, this=value)
                
            return select

        # def _parse_error_clause(self):
        #     if self._match_text_seq("ERROR"):
        #         value = self._parse_primary()
        #         if not value:
        #             if not isinstance(value, (int, float)):
        #                 self.error("Unidentified Literal after ERROR")
        #             self.error("Expected value after ERROR")
                
        #         return self.expression(Error, this=value)
        #     return None
        
        # def _parse_confidence_clause(self):
        #     if self._match_text_seq("CONFIDENCE"):
        #         value = self._parse_primary()
        #         if not value:
        #             if not isinstance(value, (int, float)):
        #                 self.error("Unidentified Literal after CONFIDENCE")
        #             self.error("Expected value after CONFIDENCE")
                
        #         return self.expression(Confidence, this=value)
        #     return None

    class Generator(Generator):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.TYPE_MAPPING[Error] = self.error_sql
            self.TYPE_MAPPING[Confidence] = self.confidence_sql
        
        def error_sql(self, expression: Error) -> str:
            return f"ERROR {self.sql(expression, 'this')}"
        
        def confidence_sql(self, expression: Confidence) -> str:
            return f"CONFIDENCE {self.sql(expression, 'this')}"

        def select_sql(self, expression: exp.Select) -> str:
            sql = super().select_sql(expression)
            error_clause = expression.args.get('error')
            if error_clause:
                return f"{sql} {self.sql(error_clause)}"
            
            confidence_clause = expression.args.get('confidence')
            if confidence_clause:
                return f"{sql} {self.sql(confidence_clause)}"
            return sql