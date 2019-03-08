import pandas as pd

from matching import Matcher

vacancies_parts = pd.read_csv('../data/new/vacancy_parts_for_matching.csv')[:100]
profstandards_parts = pd.read_csv('../data/new/profstandard_parts_for_matching.csv')

matcher = Matcher()
result = matcher.match_parts(vacancies_parts, profstandards_parts)
result.to_csv('../data/new/sim_result.csv', index=False)
