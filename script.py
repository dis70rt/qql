from utils.db_connect import PSQL, QQLEnv
from execute import execute_query

env = QQLEnv()
db = PSQL(dbname='postgres' , 
          user='ethernode', 
          password='sd1312004', 
          host='127.0.0.1', 
          port='5432')

sql = """SELECT APPROX pizza_type, AVG(price) FROM pizza_orders GROUP BY pizza_type ERROR 0.05 CONFIDENCE 0.95;"""
res = execute_query(sql, db)
print(res)
db.close()