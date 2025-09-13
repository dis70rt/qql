import psycopg2
import os
from pathlib import Path

class PSQL:
    def __init__(self, dbname, user, password, host, port):
        self.conn = psycopg2.connect(
            dbname=dbname,
            user=user,
            password=password,
            host=host,
            port=port
        )
        self.cur = self.conn.cursor()

    def execute(self, query, params=None):
        try:
            self.conn.rollback()
            self.cur.execute(query, params or ())
            self.conn.commit()
            return self.cur
        except:
            self.conn.rollback()
            raise

    def close(self):
        self.cur.close()
        self.conn.close()


class QQLEnv:
    ENV_FILE = Path.home() / ".qql_env"

    def __init__(self):
        self.username = None
        self.password = None
        self.dbname = None
        self.host = "localhost"
        self.port = 5432
        self.load()

    def load(self):
        if self.ENV_FILE.exists():
            with open(self.ENV_FILE) as f:
                for line in f:
                    if "=" in line:
                        key, val = line.strip().split("=", 1)
                        os.environ[key] = val

        self.username = os.environ.get("DB_USERNAME")
        self.password = os.environ.get("DB_PASSWORD")
        self.dbname = os.environ.get("DB_NAME")
        self.host = os.environ.get("DB_HOST", "localhost")
        port = os.environ.get("DB_PORT")
        self.port = int(port) if port else 5432

    def save(self, username, password, dbname=None, host=None, port=None):
        dbname = dbname or self.dbname
        host = host or self.host
        port = port or self.port

        with open(self.ENV_FILE, "w") as f:
            f.write(f"DB_USERNAME={username}\n")
            f.write(f"DB_PASSWORD={password}\n")
            if dbname:
                f.write(f"DB_NAME={dbname}\n")
            if host:
                f.write(f"DB_HOST={host}\n")
            if port:
                f.write(f"DB_PORT={port}\n")

        os.environ["DB_USERNAME"] = username
        os.environ["DB_PASSWORD"] = password
        if dbname:
            os.environ["DB_NAME"] = dbname
        if host:
            os.environ["DB_HOST"] = host
        if port:
            os.environ["DB_PORT"] = str(port)

        self.username = username
        self.password = password
        self.dbname = dbname
        self.host = host
        self.port = port
