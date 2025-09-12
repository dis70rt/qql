import cmd

class QQLShell(cmd.Cmd):
    intro = "Type exit to quit.\n"
    prompt = "qql> "
    buffer = ""

    def default(self, line):
        self.buffer += " " + line.strip()

        if self.buffer.strip().endswith(";"):
            statement = self.buffer.strip()[:-1]
            self.execute_sql(statement)
            self.buffer = ""

    def execute_sql(self, statement):
        # DO HERE
        print(f"Executing SQL: {statement}")
        
    def do_exit(self, arg):
        return True
    do_quit = do_exit

if __name__ == "__main__":
    QQLShell().cmdloop()
