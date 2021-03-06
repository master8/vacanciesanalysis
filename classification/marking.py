import pandas as pd

# marks = pd.read_csv("../data/new/standard_marks.csv", header=0)
# data = pd.read_csv("../data/new/marked_vacancies_hh_all_131018.csv", header=0, sep='|')
# data[data.requirements.notnull() & data.duties.notnull()].standard_mark.value_counts()
# data.to_csv("../data/new/marked_vacancies_hh_all_131018.csv", index=False, sep='|')


def mark_corpus(data: pd.DataFrame) -> pd.DataFrame:

    data['standard_mark'] = 0

    # Разметка DBA
    data.loc[
        # data.requirements.notnull()
        # & data.duties.notnull()
        data.specializations.str.contains('1\.420')
        & ~data.specializations.str.contains('1\.221')
        & data.name.str.contains('администратор', case=False)
        & ~data.name.str.contains('системный', case=False)
        & (data.name.str.contains('баз данных', case=False)
           | data.name.str.contains('DBA', case=False)
           | data.name.str.contains('Oracle', case=False)
           | data.name.str.contains('MS SQL', case=False)
           | data.name.str.contains('MySQL', case=False)
           | data.name.str.contains('PostgreSQL', case=False)
           | data.name.str.contains('БД', case=False)
           ), 'standard_mark'] = 1

    # Разметка Технический писатель
    data.loc[
        # data.requirements.notnull()
        # & data.duties.notnull()
        (data.standard_mark == 0)
        & data.specializations.str.contains('1.296')
        & (data.name.str.contains('Технический писатель', case=False)
           | data.name.str.contains('технической документации', case=False)
           | data.name.str.contains('Technical Writer', case=False)
           ), 'standard_mark'] = 7

    # Разметка Системный программист
    data.loc[
        # data.requirements.notnull()
        # & data.duties.notnull()
        (data.standard_mark == 0)
        & (data.name.str.contains('Системный программист', case=False)
           | data.name.str.contains('Системный разработчик', case=False)
           ), 'standard_mark'] = 12

    # Разметка Архитектор программного обеспечения
    data.loc[
        # data.requirements.notnull()
        # & data.duties.notnull()
        (data.standard_mark == 0)
        & data.name.str.contains('Архитектор', case=False)
        , 'standard_mark'] = 17

    # Разметка Системный аналитик
    data.loc[
        # data.requirements.notnull()
        # & data.duties.notnull()
        (data.standard_mark == 0)
        & data.name.str.contains('аналитик', case=False)
        & data.specializations.str.contains('1.25')
        , 'standard_mark'] = 8

    # Разметка Специалист по дизайну
    data.loc[
        # data.requirements.notnull()
        # & data.duties.notnull()
        (data.standard_mark == 0)
        & data.specializations.str.contains('1.30')
        , 'standard_mark'] = 9

    # Разметка Специалист по интеграции
    data.loc[
        # data.requirements.notnull()
        # & data.duties.notnull()
        (data.standard_mark == 0)
        & ~data.name.str.contains('Менеджер', case=False)
        & ~data.name.str.contains('Руководитель', case=False)
        & ~data.name.str.contains('Директор', case=False)
        & (data.name.str.contains('внедрение', case=False)
           | data.name.str.contains('внедрению', case=False)
           | data.name.str.contains('внедрения', case=False)
           | data.name.str.contains('интеграции', case=False)
           ), 'standard_mark'] = 21

    # Разметка телекоммуникации
    data.loc[
        # data.requirements.notnull()
        # & data.duties.notnull()
        (data.standard_mark == 0)
        & (data.specializations.str.contains('1.295')
           | data.specializations.str.contains('1.277')
           )
        & ~data.specializations.str.contains('1.10')
        & ~data.specializations.str.contains('1.9')
        & (data.name.str.contains('Инженер-проектировщик', case=False)
           | data.name.str.contains('Инженер проектировщик', case=False)
           ), 'standard_mark'] = 20

    # Разметка Разработчик Web и мультимедийных приложений
    data.loc[
        # data.requirements.notnull()
        # & data.duties.notnull()
        (data.standard_mark == 0)
        & (data.specializations.str.contains('1\.9')
           | data.specializations.str.contains('1\.10')
           )
        & ~data.specializations.str.contains('1\.475')
        & ~data.specializations.str.contains('1\.211')
        & ~data.specializations.str.contains('1\.161')
        & ~(data.name.str.contains('Оператор видеонаблюдения', case=False)
            | data.name.str.contains('Контент-менеджер', case=False)
            | data.name.str.contains('администратор', case=False)
            | data.name.str.contains('Дизайнер', case=False)
            | data.name.str.contains('Руководитель', case=False)
            | data.name.str.contains('Android', case=False)
            | data.name.str.contains('iOS', case=False)
            | data.name.str.contains('маркетолог', case=False)
            | data.name.str.contains('Тестировщик', case=False)
            | data.name.str.contains('Менеджер', case=False)
            | data.name.str.contains('мобильных', case=False)
            | data.name.str.contains('Начальник', case=False)
            | data.name.str.contains('C\+\+', case=False)
            | data.name.str.contains('Team Lead', case=False)
            | data.name.str.contains('QA', case=False)
            | data.name.str.contains('UI', case=False)
            | data.name.str.contains('тестирования', case=False)
            | data.name.str.contains('manager', case=False)
            ), 'standard_mark'] = 19

    # Разметка Специалист по информационным системам
    data.loc[
        # data.requirements.notnull()
        # & data.duties.notnull()
        (data.standard_mark == 0)
        & (data.specializations.str.contains('1\.50')
           | data.specializations.str.contains('1\.536')
           )
        & ~data.specializations.str.contains('1\.221')
        & ~data.name.str.contains('программист', case=False)
        & ~data.name.str.contains('Системный администратор', case=False)
        & ~data.name.str.contains('поддержк*', case=False)
        & (data.name.str.contains('специалист', case=False)
           | data.name.str.contains('консультант', case=False)
           ), 'standard_mark'] = 2

    # Разметка Менеджер продуктов в области информационных технологий
    data.loc[
        # data.requirements.notnull()
        # & data.duties.notnull()
        (data.standard_mark == 0)
        & data.name.str.contains('Менеджер .*продукт.*', case=False)
        , 'standard_mark'] = 3

    # Разметка Руководитель проектов в области информационных технологий
    data.loc[
        # data.requirements.notnull()
        # & data.duties.notnull()
        (data.standard_mark == 0)
        & data.specializations.str.contains('1\.327')
        & ~data.specializations.str.contains('1\.221')
        & data.name.str.contains('.*проект.*', case=False)
        & ~data.name.str.contains('офис*', case=False)
        & ~data.name.str.contains('отдел*', case=False)
        & (data.name.str.contains('Руководитель', case=False)
           | data.name.str.contains('менеджер', case=False)
           | data.name.str.contains('Project Manager', case=False)
           ), 'standard_mark'] = 5

    # Разметка Менеджер по информационным технологиям
    data.loc[
        # data.requirements.notnull()
        # & data.duties.notnull()
        (data.standard_mark == 0)
        & data.specializations.str.contains('1\.3')
        & ~data.specializations.str.contains('1\.395')
        & ~data.specializations.str.contains('1\.359')
        & ~data.specializations.str.contains('1\.327')
        & ~data.specializations.str.contains('1\.30')
        & (data.name.str.contains('Начальник', case=False)
           | data.name.str.contains('подраздилен.*', case=False)
           | data.name.str.contains('департамент.*', case=False)
           | data.name.str.contains('директор', case=False)
           | data.name.str.contains('CTO', case=False)
           | data.name.str.contains('CIO', case=False)
           ), 'standard_mark'] = 4

    # Разметка Руководитель разработки программного обеспечения
    data.loc[
        # data.requirements.notnull()
        # & data.duties.notnull()
        (data.standard_mark == 0)
        & data.name.str.contains('Руководитель разработки', case=False)
        , 'standard_mark'] = 6

    data.loc[
        # data.requirements.notnull()
        # & data.duties.notnull()
        (data.standard_mark == 0)
        & data.name.str.contains('Руководитель .*разработки', case=False)
        & ~data.name.str.contains('проект', case=False)
        , 'standard_mark'] = 6

    # Разметка Системный администратор информационно-коммуникационных систем
    data.loc[
        # data.requirements.notnull()
        # & data.duties.notnull()
        (data.standard_mark == 0)
        & data.name.str.contains('Системный администратор', case=False)
        , 'standard_mark'] = 10

    # Разметка Специалист по администрированию сетевых устройств информационно-коммуникационных систем
    data.loc[
        # data.requirements.notnull()
        # & data.duties.notnull()
        (data.standard_mark == 0)
        & ~data.specializations.str.contains('1\.221')
        & data.name.str.contains('сетев.*', case=False)
        & ~data.name.str.contains('безопасн.*', case=False)
        , 'standard_mark'] = 11

    # Разметка Специалист по тестированию в области информационных технологий
    data.loc[
        # data.requirements.notnull()
        # & data.duties.notnull()
        (data.standard_mark == 0)
        & ~data.specializations.str.contains('1\.221')
        & (data.name.str.contains('Специалист по тестированию', case=False)
           | data.name.str.contains('Тестировщик', case=False)
           ), 'standard_mark'] = 18

    # Разметка Специалист по информационным ресурсам
    data.loc[
        # data.requirements.notnull()
        # & data.duties.notnull()
        (data.standard_mark == 0)
        & data.specializations.str.contains('1\.116')
        & data.name.str.contains('контент', case=False)
        , 'standard_mark'] = 16

    # Разметка Инженер-радиоэлектронщик
    data.loc[
        # data.requirements.notnull()
        # & data.duties.notnull()
        (data.standard_mark == 0)
        & data.specializations.str.contains('1\.82')
        & ~data.specializations.str.contains('1\.221')
        & data.name.str.contains('радио.*', case=False)
        & ~data.name.str.contains('писатель', case=False)
        , 'standard_mark'] = 15

    # Разметка Программист
    data.loc[
        # data.requirements.notnull()
        # & data.duties.notnull()
        (data.standard_mark == 0)
        & data.specializations.str.contains('1\.221')
        & (data.name.str.contains('Программист', case=False)
           | data.name.str.contains('разработчик', case=False)
           ), 'standard_mark'] = 14

    return data


def clean_label(label: str):
    t = label.split(',')
    t = list(map(str.strip, t))

    if '' in t:
        t.remove('')

    if '13' in t:
        t.remove('13')

    return ','.join(t)


def mark_corpus_multi_labels(data: pd.DataFrame) -> pd.DataFrame:

    data['labels'] = ''

    # Разметка DBA
    data.loc[
        data.specializations.str.contains('1\.420')
        & ~data.specializations.str.contains('1\.221')
        & data.name.str.contains('администратор', case=False)
        & ~data.name.str.contains('системный', case=False)
        & (data.name.str.contains('баз данных', case=False)
           | data.name.str.contains('DBA', case=False)
           | data.name.str.contains('Oracle', case=False)
           | data.name.str.contains('MS SQL', case=False)
           | data.name.str.contains('MySQL', case=False)
           | data.name.str.contains('PostgreSQL', case=False)
           | data.name.str.contains('БД', case=False)
           ), 'labels'] = '1' + ',' + data.labels

    # Разметка Технический писатель
    data.loc[
        data.specializations.str.contains('1.296')
        & (data.name.str.contains('Технический писатель', case=False)
           | data.name.str.contains('технической документации', case=False)
           | data.name.str.contains('Technical Writer', case=False)
           ), 'labels'] = '7' + ',' + data.labels

    # Разметка Системный программист
    data.loc[
        (data.name.str.contains('Системный программист', case=False)
           | data.name.str.contains('Системный разработчик', case=False)
           ), 'labels'] = '12' + ',' + data.labels

    # Разметка Архитектор программного обеспечения
    data.loc[
        data.name.str.contains('Архитектор', case=False)
        , 'labels'] = '17' + ',' + data.labels

    # Разметка Системный аналитик
    data.loc[
        data.name.str.contains('аналитик', case=False)
        & data.specializations.str.contains('1.25')
        , 'labels'] = '8' + ',' + data.labels

    # Разметка Специалист по дизайну
    data.loc[
        data.specializations.str.contains('1.30')
        , 'labels'] = '9' + ',' + data.labels

    # Разметка Специалист по интеграции
    data.loc[
        ~data.name.str.contains('Менеджер', case=False)
        & ~data.name.str.contains('Руководитель', case=False)
        & ~data.name.str.contains('Директор', case=False)
        & (data.name.str.contains('внедрение', case=False)
           | data.name.str.contains('внедрению', case=False)
           | data.name.str.contains('внедрения', case=False)
           | data.name.str.contains('интеграции', case=False)
           ), 'labels'] = '21' + ',' + data.labels

    # Разметка телекоммуникации
    data.loc[
        (data.specializations.str.contains('1.295')
           | data.specializations.str.contains('1.277')
           )
        & ~data.specializations.str.contains('1.10')
        & ~data.specializations.str.contains('1.9')
        & (data.name.str.contains('Инженер-проектировщик', case=False)
           | data.name.str.contains('Инженер проектировщик', case=False)
           ), 'labels'] = '20' + ',' + data.labels

    # Разметка Разработчик Web и мультимедийных приложений
    data.loc[
        (data.specializations.str.contains('1\.9')
           | data.specializations.str.contains('1\.10')
           )
        & ~data.specializations.str.contains('1\.475')
        & ~data.specializations.str.contains('1\.211')
        & ~data.specializations.str.contains('1\.161')
        & ~(data.name.str.contains('Оператор видеонаблюдения', case=False)
            | data.name.str.contains('Контент-менеджер', case=False)
            | data.name.str.contains('администратор', case=False)
            | data.name.str.contains('Дизайнер', case=False)
            | data.name.str.contains('Руководитель', case=False)
            | data.name.str.contains('Android', case=False)
            | data.name.str.contains('iOS', case=False)
            | data.name.str.contains('маркетолог', case=False)
            | data.name.str.contains('Тестировщик', case=False)
            | data.name.str.contains('Менеджер', case=False)
            | data.name.str.contains('мобильных', case=False)
            | data.name.str.contains('Начальник', case=False)
            | data.name.str.contains('C\+\+', case=False)
            | data.name.str.contains('Team Lead', case=False)
            | data.name.str.contains('QA', case=False)
            | data.name.str.contains('UI', case=False)
            | data.name.str.contains('тестирования', case=False)
            | data.name.str.contains('manager', case=False)
            ), 'labels'] = '19' + ',' + data.labels

    # Разметка Специалист по информационным системам
    data.loc[
        (data.specializations.str.contains('1\.50')
           | data.specializations.str.contains('1\.536')
           )
        & ~data.specializations.str.contains('1\.221')
        & ~data.name.str.contains('программист', case=False)
        & ~data.name.str.contains('Системный администратор', case=False)
        & ~data.name.str.contains('поддержк*', case=False)
        & (data.name.str.contains('специалист', case=False)
           | data.name.str.contains('консультант', case=False)
           ), 'labels'] = '2' + ',' + data.labels

    # Разметка Менеджер продуктов в области информационных технологий
    data.loc[
        data.name.str.contains('Менеджер .*продукт.*', case=False)
        , 'labels'] = '3' + ',' + data.labels

    # Разметка Руководитель проектов в области информационных технологий
    data.loc[
        data.specializations.str.contains('1\.327')
        & ~data.specializations.str.contains('1\.221')
        & data.name.str.contains('.*проект.*', case=False)
        & ~data.name.str.contains('офис*', case=False)
        & ~data.name.str.contains('отдел*', case=False)
        & (data.name.str.contains('Руководитель', case=False)
           | data.name.str.contains('менеджер', case=False)
           | data.name.str.contains('Project Manager', case=False)
           ), 'labels'] = '5' + ',' + data.labels

    # Разметка Менеджер по информационным технологиям
    data.loc[
        data.specializations.str.contains('1\.3')
        & ~data.specializations.str.contains('1\.395')
        & ~data.specializations.str.contains('1\.359')
        & ~data.specializations.str.contains('1\.327')
        & ~data.specializations.str.contains('1\.30')
        & (data.name.str.contains('Начальник', case=False)
           | data.name.str.contains('подраздилен.*', case=False)
           | data.name.str.contains('департамент.*', case=False)
           | data.name.str.contains('директор', case=False)
           | data.name.str.contains('CTO', case=False)
           | data.name.str.contains('CIO', case=False)
           ), 'labels'] = '4' + ',' + data.labels

    # Разметка Руководитель разработки программного обеспечения

    data.loc[
        data.name.str.contains('Руководитель .*разработки', case=False)
        & ~data.name.str.contains('проект', case=False)
        , 'labels'] = '6' + ',' + data.labels

    # Разметка Системный администратор информационно-коммуникационных систем
    data.loc[
        data.name.str.contains('Системный администратор', case=False)
        , 'labels'] = '10' + ',' + data.labels

    # Разметка Специалист по администрированию сетевых устройств информационно-коммуникационных систем
    data.loc[
        ~data.specializations.str.contains('1\.221')
        & data.name.str.contains('сетев.*', case=False)
        & ~data.name.str.contains('безопасн.*', case=False)
        , 'labels'] = '11' + ',' + data.labels

    # Разметка Специалист по тестированию в области информационных технологий
    data.loc[
        ~data.specializations.str.contains('1\.221')
        & (data.name.str.contains('Специалист по тестированию', case=False)
           | data.name.str.contains('Тестировщик', case=False)
           ), 'labels'] = '18' + ',' + data.labels

    # Разметка Специалист по информационным ресурсам
    data.loc[
        data.specializations.str.contains('1\.116')
        & data.name.str.contains('контент', case=False)
        , 'labels'] = '16' + ',' + data.labels

    # Разметка Инженер-радиоэлектронщик
    data.loc[
        data.specializations.str.contains('1\.82')
        & ~data.specializations.str.contains('1\.221')
        & data.name.str.contains('радио.*', case=False)
        & ~data.name.str.contains('писатель', case=False)
        , 'labels'] = '15' + ',' + data.labels

    # Разметка Программист
    data.loc[
        data.specializations.str.contains('1\.221')
        & (data.name.str.contains('Программист', case=False)
           | data.name.str.contains('разработчик', case=False)
           ), 'labels'] = '14' + ',' + data.labels

    data.labels = data.labels.apply(clean_label)

    return data


def merge_marking(corpus_original_name: str, corpus_edited_name: str, corpus_result_name: str):
    data = pd.read_csv("../data/new/" + corpus_original_name, header=0, index_col='id')
    ed = pd.read_csv("../data/new/" + corpus_edited_name, header=0, index_col='id')

    # ed = ed.drop(columns=['name', 'duties', 'requirements',
    #    'all_description', 'has_duties', 'has_requirements', 'standard_mark',
    #    'proba_true_w2v', 'pred_mark_w2v', 'proba_pred_w2v', 'proba_true_tfidf',
    #    'pred_mark_tfidf', 'proba_pred_tfidf'])

    ed = ed.drop(columns=['name', 'all_description', 'labels',
       'w2v', 'tfidf', 'max_wrong', 'm1', 'm2',
       'm3', 'm4', 'm5', 'm6', 'm7', 'm8', 'm9', 'm10', 'm11', 'm12', 'm14',
       'm15', 'm16', 'm17', 'm18', 'm19', 'm20', 'm21'])

    co = pd.merge(data, ed, left_index=True, right_index=True, how='outer', suffixes=('', '_y'))

    co[co.main_label != 0].main_label.count()
    co[co.main_label_y != 0].main_label_y.count()

    co['main_label_y'] = co['main_label_y'].fillna(0).astype('int')
    co.loc[co.main_label_y != 0, 'main_label'] = co.main_label_y

    co['additional_labels_y'] = co['additional_labels_y'].fillna('0,0,0')
    co.loc[co.additional_labels_y != '0,0,0', 'additional_labels'] = co.additional_labels_y

    co['editor_name_y'] = co['editor_name_y'].fillna('')
    co.loc[co.editor_name_y != '', 'editor_name'] = co.editor_name_y

    co['comments_y'] = co['comments_y'].fillna('')
    co.loc[co.comments_y != '', 'comments'] = co.comments_y

    co = co[co.main_label != -1]
    co = co[co.main_label != 13]

    co.loc[co.main_label != 0, 'standard_mark'] = co.main_label
    co = co.drop(columns=['main_label_y', 'additional_labels_y', 'editor_name_y', 'comments_y'])

    co['main_label'] = co['main_label'].fillna(0).astype('int')
    co['additional_labels'] = co['additional_labels'].fillna('')
    co.loc[co.additional_labels == '0,0,0', 'additional_labels'] = ''
    co['editor_name'] = co['editor_name'].fillna('')
    co['comments'] = co['comments'].fillna('')

    co.loc[co.main_label != 0, 'labels'] = co.main_label.apply(str) + ',' + co.additional_labels

    co.labels = co.labels.apply(clean_label)

    co.to_csv("../data/new/" + corpus_result_name, index_label='id')
