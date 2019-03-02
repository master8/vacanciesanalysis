import pandas as pd
import razdel


co = pd.read_csv('data/new/all_duties_and_requirements.csv')


def split_by_tilda_and_join(text):
    text = text.split('~')
    text = '\n'.join(text)
    return text


def split_by_sentence(text):
    l = [_.text for _ in list(razdel.sentenize(split_by_tilda_and_join(text)))]
    return '\n'.join(l)


co['duties'] = co.duties.astype('str')
co['duties_s'] = co.duties.apply(lambda x: split_by_sentence(x))
co.index = co.id

t = co['duties_s'].str.split('\n').apply(pd.Series, 1).stack()
t.index = t.index.droplevel(-1)
t.name = 'text'
d = pd.DataFrame(t)
d['vacancy_id'] = d.index
d['type_id'] = 1


co['requirements'] = co.requirements.astype('str')
co['requirements_s'] = co.requirements.apply(lambda x: split_by_sentence(x))
co.index = co.id

t = co['requirements_s'].str.split('\n').apply(pd.Series, 1).stack()
t.index = t.index.droplevel(-1)
t.name = 'text'
r = pd.DataFrame(t)
r['vacancy_id'] = r.index
r['type_id'] = 2

re = pd.concat([d, r])
re['id'] = range(1, 3690918, 1)
re.describe()
