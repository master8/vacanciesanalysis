from datetime import datetime

import pandas as pd
import pymystem3
import logging

# logging.basicConfig(filename='main.log', level=logging.INFO)
# logging.warning('Start main!')

from classification.experiments.grboost import GradientBoostingExperiments
from classification.experiments.knn import KNeighborsExperiments
from classification.experiments.logreg import LogisticRegressionExperiments
from classification.experiments.svc import SVCExperiments
from classification.experiments.voting import VotingExperiments
from classification.tokenization import Tokenizer, TokensProvider
from classification.vectorization import Vectorizer, VectorsProvider

pymystem3.mystem.MYSTEM_DIR = "/home/mluser/anaconda3/envs/master8_env/.local/bin"
pymystem3.mystem.MYSTEM_BIN = "/home/mluser/anaconda3/envs/master8_env/.local/bin/mystem"

# Tokenizer().tokenize()

# vectorizer = Vectorizer()
# vectorizer.vectorize_with_tfidf()
# vectorizer.vectorize_with_w2v()
# vectorizer.vectorize_with_w2v_tfidf()
# vectorizer.vectorize_with_tfidf_wshingles()
# vectorizer.vectorize_with_tfidf_ngrams()
# vectorizer.vectorize_with_w2v_big()
# vectorizer.vectorize_with_w2v_old()


# logreg = LogisticRegressionExperiments()
# logreg.make_use_tfidf()
# logreg.make_use_w2v()
# logreg.make_use_w2v_with_tfidf()
# logreg.make_use_tfidf_wshingles()
# logreg.make_use_tfidf_ngrams()
# logreg.make_use_w2v_big()
# logreg.make_use_w2v_old()

# knn = KNeighborsExperiments()
# knn.make_use_tfidf()
# knn.make_use_w2v()
# knn.make_use_w2v_with_tfidf()
# knn.make_use_tfidf_wshingles()
# knn.make_use_tfidf_ngrams()
# knn.make_use_w2v_big()
# knn.make_use_w2v_old()

# svc = SVCExperiments()
# svc.make_use_tfidf()
# svc.make_use_w2v()
# svc.make_use_w2v_with_tfidf()
# svc.make_use_tfidf_wshingles()
# svc.make_use_tfidf_ngrams()
# svc.make_use_w2v_big()
# svc.make_use_w2v_old()

# voting = VotingExperiments()
# voting.make_use_tfidf()
# voting.make_use_w2v()
# voting.make_use_w2v_with_tfidf()
# voting.make_use_tfidf_wshingles()
# voting.make_use_tfidf_ngrams()
# voting.make_use_w2v_big()

# grboost = GradientBoostingExperiments()
# grboost.make_use_tfidf()
# grboost.make_use_w2v()
# grboost.make_use_w2v_with_tfidf()
# grboost.make_use_tfidf_wshingles()
# grboost.make_use_tfidf_ngrams()
# grboost.make_use_w2v_big()

data = pd.read_csv("../data/new/marked_vacancies_hh_all_131018.csv", header=0, sep='|')
# data.to_csv("../data/new/marked_vacancies_hh_all_131018.csv", index=False, sep='|')

# data[data.requirements.notnull() & data.duties.notnull()].standard_mark.value_counts()
# data.loc[data.id == 20000000, 'custom_mark'] = 1
# data[data.name.str.contains('QA')].name[:50]
# data[data.specializations.str.contains('1.137')].name[:50]

# data[data.specializations.str.contains('1.420') & data.requirements.notnull()].name.describe()
# data[data.specializations.str.contains('1.420') & ~data.specializations.str.contains('1.221') & data.requirements.notnull()][['name', 'specializations']][:10]

# Фильтрация для Системный администратор информационно-коммуникационных систем
# var = data[data.requirements.notnull()
#            & data.duties.notnull()
#            & (data.standard_mark == 0)
#            & data.name.str.contains('Системный администратор', case=False)
#            ][['name', 'specializations']][:50]
#
# Разметка Системный администратор информационно-коммуникационных систем
# data.loc[
#     data.requirements.notnull()
#     & data.duties.notnull()
#     & (data.standard_mark == 0)
#     & data.name.str.contains('Системный администратор', case=False)
#     , 'standard_mark'] = 10


# Фильтрация для Руководитель разработки программного обеспечения
# var = data[data.requirements.notnull()
#            & data.duties.notnull()
#            & (data.standard_mark == 0)
#            & data.name.str.contains('Руководитель разработки', case=False)
#            ][['name', 'specializations']][:50]
#
# Разметка Руководитель разработки программного обеспечения
# data.loc[
#     data.requirements.notnull()
#     & data.duties.notnull()
#     & (data.standard_mark == 0)
#     & data.name.str.contains('Руководитель разработки', case=False)
#     , 'standard_mark'] = 6


# Фильтрация для Менеджер по информационным технологиям
# var = data[data.requirements.notnull()
#            & data.duties.notnull()
#            & (data.standard_mark == 0)
#            & data.specializations.str.contains('1\.3')
#            & ~data.specializations.str.contains('1\.395')
#            & ~data.specializations.str.contains('1\.359')
#            & ~data.specializations.str.contains('1\.327')
#            & ~data.specializations.str.contains('1\.30')
#            & (data.name.str.contains('Начальник', case=False)
#               | data.name.str.contains('подраздилен.*', case=False)
#               | data.name.str.contains('департамент.*', case=False)
#               | data.name.str.contains('директор', case=False)
#               | data.name.str.contains('CTO', case=False)
#               | data.name.str.contains('CIO', case=False)
#               )
#            ][['name', 'specializations']][:50]
#
# Разметка Менеджер по информационным технологиям
# data.loc[
#     data.requirements.notnull()
#     & data.duties.notnull()
#     & (data.standard_mark == 0)
#     & data.specializations.str.contains('1\.3')
#     & ~data.specializations.str.contains('1\.395')
#     & ~data.specializations.str.contains('1\.359')
#     & ~data.specializations.str.contains('1\.327')
#     & ~data.specializations.str.contains('1\.30')
#     & (data.name.str.contains('Начальник', case=False)
#        | data.name.str.contains('подраздилен.*', case=False)
#        | data.name.str.contains('департамент.*', case=False)
#        | data.name.str.contains('директор', case=False)
#        | data.name.str.contains('CTO', case=False)
#        | data.name.str.contains('CIO', case=False)
#        ), 'standard_mark'] = 4


# Фильтрация для Руководитель проектов в области информационных технологий
# var = data[data.requirements.notnull()
#            & data.duties.notnull()
#            & (data.standard_mark == 0)
#            & data.specializations.str.contains('1\.327')
#            & ~data.specializations.str.contains('1\.221')
#            & data.name.str.contains('.*проект.*', case=False)
#            & ~data.name.str.contains('офис*', case=False)
#            & ~data.name.str.contains('отдел*', case=False)
#            & (data.name.str.contains('Руководитель', case=False)
#               | data.name.str.contains('менеджер', case=False)
#               | data.name.str.contains('Project Manager', case=False)
#               )
#            ][['name', 'specializations']][:50]
#
# Разметка Руководитель проектов в области информационных технологий
# data.loc[
#     data.requirements.notnull()
#     & data.duties.notnull()
#     & (data.standard_mark == 0)
#     & data.specializations.str.contains('1\.327')
#     & ~data.specializations.str.contains('1\.221')
#     & data.name.str.contains('.*проект.*', case=False)
#     & ~data.name.str.contains('офис*', case=False)
#     & ~data.name.str.contains('отдел*', case=False)
#     & (data.name.str.contains('Руководитель', case=False)
#        | data.name.str.contains('менеджер', case=False)
#        | data.name.str.contains('Project Manager', case=False)
#        ), 'standard_mark'] = 5


# Фильтрация для Менеджер продуктов в области информационных технологий
# var = data[data.requirements.notnull()
#            & data.duties.notnull()
#            & (data.standard_mark == 0)
#            & data.name.str.contains('Менеджер .*продукт.*', case=False)
#            ][['name', 'specializations']][:50]
#
# Разметка Менеджер продуктов в области информационных технологий
# data.loc[
#     data.requirements.notnull()
#     & data.duties.notnull()
#     & (data.standard_mark == 0)
#     & data.name.str.contains('Менеджер .*продукт.*', case=False)
#     , 'standard_mark'] = 3


# Фильтрация для Специалист по информационным системам
# var = data[data.requirements.notnull()
#            & data.duties.notnull()
#            & (data.standard_mark == 0)
#            & (data.specializations.str.contains('1\.50')
#               | data.specializations.str.contains('1\.536')
#               )
#            & ~data.specializations.str.contains('1\.221')
#            & ~data.name.str.contains('программист', case=False)
#            & ~data.name.str.contains('Системный администратор', case=False)
#            & ~data.name.str.contains('поддержк*', case=False)
#            & (data.name.str.contains('специалист', case=False)
#               | data.name.str.contains('консультант', case=False)
#               )
#            ][['name', 'specializations']][:50]
#
# Разметка Специалист по информационным системам
# data.loc[
#     data.requirements.notnull()
#     & data.duties.notnull()
#     & (data.standard_mark == 0)
#     & (data.specializations.str.contains('1\.50')
#        | data.specializations.str.contains('1\.536')
#        )
#     & ~data.specializations.str.contains('1\.221')
#     & ~data.name.str.contains('программист', case=False)
#     & ~data.name.str.contains('Системный администратор', case=False)
#     & ~data.name.str.contains('поддержк*', case=False)
#     & (data.name.str.contains('специалист', case=False)
#        | data.name.str.contains('консультант', case=False)
#        ), 'standard_mark'] = 2


# Фильтрация для Разработчик Web и мультимедийных приложений
# var = data[data.requirements.notnull()
#            & data.duties.notnull()
#            & (data.standard_mark == 0)
#            & (data.specializations.str.contains('1\.9')
#               | data.specializations.str.contains('1\.10')
#               )
#            & ~data.specializations.str.contains('1\.475')
#            & ~data.specializations.str.contains('1\.211')
#            & ~data.specializations.str.contains('1\.161')
#            & ~(data.name.str.contains('Оператор видеонаблюдения', case=False)
#               | data.name.str.contains('Контент-менеджер', case=False)
#               | data.name.str.contains('администратор', case=False)
#               | data.name.str.contains('Дизайнер', case=False)
#               | data.name.str.contains('Руководитель', case=False)
#               | data.name.str.contains('Android', case=False)
#               | data.name.str.contains('iOS', case=False)
#               | data.name.str.contains('маркетолог', case=False)
#               | data.name.str.contains('Тестировщик', case=False)
#               | data.name.str.contains('Менеджер', case=False)
#               | data.name.str.contains('мобильных', case=False)
#               | data.name.str.contains('Начальник', case=False)
#               | data.name.str.contains('C\+\+', case=False)
#               | data.name.str.contains('Team Lead', case=False)
#               | data.name.str.contains('QA', case=False)
#               | data.name.str.contains('UI', case=False)
#               | data.name.str.contains('тестирования', case=False)
#               | data.name.str.contains('manager', case=False)
#               )
#            ][['name', 'specializations']][:50]
#
# Разметка Разработчик Web и мультимедийных приложений
# data.loc[
#     data.requirements.notnull()
#     & data.duties.notnull()
#     & (data.standard_mark == 0)
#     & (data.specializations.str.contains('1\.9')
#        | data.specializations.str.contains('1\.10')
#        )
#     & ~data.specializations.str.contains('1\.475')
#     & ~data.specializations.str.contains('1\.211')
#     & ~data.specializations.str.contains('1\.161')
#     & ~(data.name.str.contains('Оператор видеонаблюдения', case=False)
#         | data.name.str.contains('Контент-менеджер', case=False)
#         | data.name.str.contains('администратор', case=False)
#         | data.name.str.contains('Дизайнер', case=False)
#         | data.name.str.contains('Руководитель', case=False)
#         | data.name.str.contains('Android', case=False)
#         | data.name.str.contains('iOS', case=False)
#         | data.name.str.contains('маркетолог', case=False)
#         | data.name.str.contains('Тестировщик', case=False)
#         | data.name.str.contains('Менеджер', case=False)
#         | data.name.str.contains('мобильных', case=False)
#         | data.name.str.contains('Начальник', case=False)
#         | data.name.str.contains('C\+\+', case=False)
#         | data.name.str.contains('Team Lead', case=False)
#         | data.name.str.contains('QA', case=False)
#         | data.name.str.contains('UI', case=False)
#         | data.name.str.contains('тестирования', case=False)
#         | data.name.str.contains('manager', case=False)
#         ), 'standard_mark'] = 19

# Фильтрация телекоммуникации
# var = data[data.requirements.notnull()
#            & data.duties.notnull()
#            & (data.standard_mark == 0)
#            & (data.specializations.str.contains('1.295')
#               | data.specializations.str.contains('1.277')
#               )
#            & ~data.specializations.str.contains('1.10')
#            & ~data.specializations.str.contains('1.9')
#            & (data.name.str.contains('Инженер-проектировщик', case=False)
#               | data.name.str.contains('Инженер проектировщик', case=False)
#               )
#            ][['name', 'specializations']][:50]
#
# Разметка телекоммуникации
# data.loc[
#     data.requirements.notnull()
#     & data.duties.notnull()
#     & (data.standard_mark == 0)
#     & (data.specializations.str.contains('1.295')
#        | data.specializations.str.contains('1.277')
#        )
#     & ~data.specializations.str.contains('1.10')
#     & ~data.specializations.str.contains('1.9')
#     & (data.name.str.contains('Инженер-проектировщик', case=False)
#        | data.name.str.contains('Инженер проектировщик', case=False)
#        ), 'standard_mark'] = 20


# Фильтрация для администратора баз данных
# var = data[data.requirements.notnull()
#            & data.duties.notnull()
#            & (data.standard_mark == 0)
#            & data.specializations.str.contains('1\.420')
#            & ~data.specializations.str.contains('1\.221')
#            & data.name.str.contains('администратор', case=False)
#            & ~data.name.str.contains('системный', case=False)
#            & (data.name.str.contains('баз данных', case=False)
#               | data.name.str.contains('DBA', case=False)
#               | data.name.str.contains('Oracle', case=False)
#               | data.name.str.contains('MS SQL', case=False)
#               | data.name.str.contains('MySQL', case=False)
#               | data.name.str.contains('PostgreSQL', case=False)
#               | data.name.str.contains('БД', case=False)
#               )
#            ][['name', 'specializations']][:50]
#
# Разметка DBA
# data.loc[
#     data.requirements.notnull()
#     & data.duties.notnull()
#     & data.specializations.str.contains('1\.420')
#     & ~data.specializations.str.contains('1\.221')
#     & data.name.str.contains('администратор', case=False)
#     & ~data.name.str.contains('системный', case=False)
#     & (data.name.str.contains('баз данных', case=False)
#        | data.name.str.contains('DBA', case=False)
#        | data.name.str.contains('Oracle', case=False)
#        | data.name.str.contains('MS SQL', case=False)
#        | data.name.str.contains('MySQL', case=False)
#        | data.name.str.contains('PostgreSQL', case=False)
#        | data.name.str.contains('БД', case=False)
#        ), 'standard_mark'] = 1
#

# фильтр Технический писатель
# var = data[data.requirements.notnull()
#            & data.duties.notnull()
#            & (data.standard_mark == 0)
#            & data.specializations.str.contains('1.296')
#            & (data.name.str.contains('Технический писатель', case=False)
#               | data.name.str.contains('технической документации', case=False)
#               | data.name.str.contains('Technical Writer', case=False)
#               )
#            ][['name', 'specializations']][:50]
#
# Разметка Технический писатель
# data.loc[
#     data.requirements.notnull()
#     & data.duties.notnull()
#     & (data.standard_mark == 0)
#     & data.specializations.str.contains('1.296')
#     & (data.name.str.contains('Технический писатель', case=False)
#        | data.name.str.contains('технической документации', case=False)
#        | data.name.str.contains('Technical Writer', case=False)
#        ), 'standard_mark'] = 7

# фильтр Системный программист
# var = data[data.requirements.notnull()
#            & data.duties.notnull()
#            & (data.standard_mark == 0)
#            & (data.name.str.contains('Системный программист', case=False)
#               | data.name.str.contains('Системный разработчик', case=False)
#               )
#            ][['name', 'specializations']][:50]

# Разметка Технический писатель
# data.loc[
#     data.requirements.notnull()
#     & data.duties.notnull()
#     & (data.standard_mark == 0)
#     & (data.name.str.contains('Системный программист', case=False)
#        | data.name.str.contains('Системный разработчик', case=False)
#        ), 'standard_mark'] = 12


# Фильтрация для Архитектор программного обеспечения
# var = data[data.requirements.notnull()
#            & data.duties.notnull()
#            & (data.standard_mark == 0)
#            & data.name.str.contains('Архитектор', case=False)
#            ][['name', 'specializations']][:50]
#
# Разметка Архитектор программного обеспечения
# data.loc[
#     data.requirements.notnull()
#     & data.duties.notnull()
#     & (data.standard_mark == 0)
#     & data.name.str.contains('Архитектор', case=False)
#     , 'standard_mark'] = 17


# Фильтрация для Системный аналитик
# var = data[data.requirements.notnull()
#            & data.duties.notnull()
#            & (data.standard_mark == 0)
#            & data.name.str.contains('аналитик', case=False)
#            & data.specializations.str.contains('1.25')
#            ][['name', 'specializations']][:50]
#
# Разметка Системный аналитик
# data.loc[
#     data.requirements.notnull()
#     & data.duties.notnull()
#     & (data.standard_mark == 0)
#     & data.name.str.contains('аналитик', case=False)
#     & data.specializations.str.contains('1.25')
#     , 'standard_mark'] = 8


# Фильтрация для Специалист по дизайну
# var = data[data.requirements.notnull()
#            & data.duties.notnull()
#            & (data.standard_mark == 0)
#            & data.specializations.str.contains('1.30')
#            ][['name', 'specializations']][:50]
#
# Разметка Специалист по дизайну
# data.loc[
#     data.requirements.notnull()
#     & data.duties.notnull()
#     & (data.standard_mark == 0)
#     & data.specializations.str.contains('1.30')
#     , 'standard_mark'] = 9


# Фильтрация Специалист по интеграции
# var = data[data.requirements.notnull()
#            & data.duties.notnull()
#            & (data.standard_mark == 0)
#            & ~data.name.str.contains('Менеджер', case=False)
#            & ~data.name.str.contains('Руководитель', case=False)
#            & ~data.name.str.contains('Директор', case=False)
#            & (data.name.str.contains('внедрение', case=False)
#               | data.name.str.contains('внедрению', case=False)
#               | data.name.str.contains('внедрения', case=False)
#               | data.name.str.contains('интеграции', case=False)
#               )
#            ][['name', 'specializations']][:50]
#
# Разметка Специалист по интеграции
# data.loc[
#     data.requirements.notnull()
#     & data.duties.notnull()
#     & (data.standard_mark == 0)
#     & ~data.name.str.contains('Менеджер', case=False)
#     & ~data.name.str.contains('Руководитель', case=False)
#     & ~data.name.str.contains('Директор', case=False)
#     & (data.name.str.contains('внедрение', case=False)
#        | data.name.str.contains('внедрению', case=False)
#        | data.name.str.contains('внедрения', case=False)
#        | data.name.str.contains('интеграции', case=False)
#        ), 'standard_mark'] = 21
