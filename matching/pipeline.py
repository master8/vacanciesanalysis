import pandas as pd
import sys

from matching import Matcher

import logging

start_n = int(sys.argv[1])

logging.basicConfig(filename='test' + str(start_n) + '.log', level=logging.INFO)
logging.warning('Start pipeline!')

logging.warning('start n = ' + str(start_n))

end_n = start_n + 95525
logging.warning('end n = ' + str(end_n))

vacancies_parts = pd.read_csv('../data/new/vacancy_parts_for_matching.csv')[start_n:end_n]
logging.warning('count =  ' + str(vacancies_parts.vacancy_part_id.count()))
profstandards_parts = pd.read_csv('../data/new/profstandard_parts_for_matching.csv')

matcher = Matcher(start_n=start_n)
result = matcher.match_parts(vacancies_parts, profstandards_parts)
result.to_csv('../data/new/sim_result' + str(start_n) + '.csv', index=False)
