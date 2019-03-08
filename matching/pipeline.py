import pandas as pd

from matching import Matcher

vacancies_parts = pd.read_csv('../data/new/vacancy_parts_for_matching.csv')[:100]
profstandards_parts = pd.read_csv('../data/new/profstandard_parts_for_matching.csv')

matcher = Matcher()
vectorized_profstandards_parts, vectorized_vacancies_parts = matcher.match_parts(vacancies_parts, profstandards_parts)
