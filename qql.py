import cmd
from utils.parser import QQL
import click
from utils.db_connect import PSQL, QQLEnv
from execute import execute_query
import timeit
from rich.console import Console
console = Console()

class QQLShell(cmd.Cmd):
    intro = "Type exit to quit.\n"
    prompt = "qql~# "
    buffer = ""
    db = None

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

    def execute_sql(self, query):
        try:
            exec_time = timeit.timeit(lambda: execute_query(query, self.db), number=1)
            console.print(f"[bold green]Execution Time:[/bold green] {exec_time*1000:.3f} ms")

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
