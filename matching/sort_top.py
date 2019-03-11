import pandas as pd
import sys

import logging

start_n = int(sys.argv[1])

logging.basicConfig(filename='top_' + str(start_n) + '.log', level=logging.INFO)
logging.warning('Start top!')

co = pd.read_csv('/home/mluser/master8_projects/pycharm_project_755/data/new/sim_result_mid' + str(start_n) + '.csv')

vpi = co.vacancy_part_id.drop_duplicates()
logging.warning('count = ' + str(vpi.count()))

a = []
i = 0
for v in vpi:
    a.append(co[co.vacancy_part_id == v].sort_values(by='similarity', ascending=False).head(n=5))
    i += 1
    if i % 3000 == 0:
        logging.warning('i = ' + str(i))

a = pd.concat(a)
a.to_csv('../data/new/top/parts' + str(start_n) + '.csv', index=False)
logging.warning('End top!')