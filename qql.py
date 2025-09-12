import cmd
import sqlglot
from utils.parser import QQL
import click
from db_connect import PSQL, QQLEnv

class QQLShell(cmd.Cmd):
    intro = "Type exit to quit.\n"
    prompt = "qql> "
    buffer = ""

    def __init__(self, username, password, dbname, host,port):
        super().__init__()
        self.username = username
        self.password = password
        self.db = PSQL(
            dbname=dbname,
            host=host,
            password=password,
            user=username,
            port=port
            )

    def default(self, line):
        self.buffer += " " + line.strip()

        if self.buffer.strip().endswith(";"):
            statement = self.buffer.strip()[:-1]
            self.execute_sql(statement)
            self.buffer = ""

    def execute_sql(self, statement):
        try:
            exp = sqlglot.parse_one(statement, read=QQL)
        except Exception as e:
            print(f"An error occurred: {e}")
        
    def do_exit(self, arg):
        return True
    do_quit = do_exit


@click.command()
@click.option("--username", help="Database username")
@click.option("--password", help="Database password")
@click.option("--dbname", help="Database name")
@click.option("--host", default="localhost", help="Database host")
@click.option("--port", default=5432, help="Database port")
def login(username, password, dbname, host, port):
    env = QQLEnv()

    username = username or env.username
    password = password or env.password
    dbname = dbname or getattr(env, "dbname", None)
    host = host or getattr(env, "host", "localhost")
    port = port or getattr(env, "port", 5432)

    if not username:
        username = click.prompt("Enter database username")
    if not password:
        password = click.prompt("Enter database password", hide_input=True)
    if not dbname:
        dbname = click.prompt("Enter database name")

    env.save(username, password)
    env.dbname = dbname
    env.host = host
    env.port = port
    env.save(username, password)

    shell = QQLShell(username, password, dbname, host, port)
    shell.cmdloop()


if __name__ == "__main__":
    login()
