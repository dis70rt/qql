# sample run script
from db_connect import PSQL, QQLEnv
from execute import execute_query
from parser import AQE_Parser
import time

env = QQLEnv()
# db = PSQL(dbname=env.dbname, user=env.username, password=env.password, host=env.host, port=env.port)
db = PSQL(dbname='postgres' , 
          user='postgres.yvaqsuzfmizddzkeksjy', 
          password='qql1234', 
          host='aws-1-ap-south-1.pooler.supabase.com', 
          port='6543')

start_ex = time.perf_counter()
sql_ex = """SELECT SUM(price) FROM pizza_orders;"""
db.execute(sql_ex)
end_ex = time.perf_counter()
print(end_ex - start_ex+1)

start = time.perf_counter()
sql = """SELECT APPROX SUM(price) FROM pizza_orders ERROR 0.05 PROB 0.98;"""
res = execute_query(sql, db)
end = time.perf_counter()
print(end - start)
# print(res)
db.close()