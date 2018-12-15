
# coding: utf-8

# In[1]:


# import
import pandas as pd
import numpy as np
# import codecs
# import os
# import pymorphy2
# from string import ascii_lowercase, digits, whitespace
# from scipy.sparse import csr_matrix
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Импорт

# In[2]:


#df_standards = pd.read_csv('../data/standards/df_standards_full_splitted.csv') #уже не надо
df_vacancies = pd.read_csv('df_vacancies.csv')
df_full = pd.read_csv("df_full.csv", sep=",", lineterminator='\n')
df_standards_prof = pd.read_csv("df_standards_prof.csv", lineterminator='\n')
full_text =pd.read_csv("full_text.csv")
samples_df = pd.read_csv('samples_df.csv')


# In[135]:


df_standards_prof['full_processed_text'] = df_standards_prof['full_text'].apply(lambda text: process_text(str(text))['lemmatized_text_pos_tags'])


# In[3]:


# df_vacancies = df_vacancies.dropna(subset=['text_item', 'type'])
# df_vacancies = df_vacancies.rename(index=str, columns={"text_item": "text"}) # уже сохранил
full_text = full_text.rename(index=str, columns={"nan_":"text"})
full_text = full_text.dropna(subset=['text'])
samples_df = samples_df.rename(index = str, columns={'vectors':'W2Vectors'})


# In[46]:


tfidf = TfidfVectorizer()
# pickle.dump(tfidf.vocabulary_,open("vectors.pkl","wb"))

#vectors_full = tfidf.fit_transform(full_text['text'])


# In[69]:


#df_standards_prof['vectors'] = tfidf.transform(df_standards_prof['processed_text'].apply(lambda x: ''.join(x))).todense().tolist()


# In[147]:


print(tfidf.get_feature_names())


# In[48]:


#pickle.dump(tfidf.vocabulary_,open("vectors.pkl","wb"))


# In[ ]:


#transformer = TfidfTransformer()
#loaded_vec = TfidfVectorizer(decode_error="replace",vocabulary=pickle.load(open("feature.pkl", "rb")))
#tfidf = transformer.fit_transform(loaded_vec.fit_transform(np.array)


# In[161]:


full_text['vectors'] = tfidf.fit_transform(full_text.text).todense().tolist() #sparse
df_standards_prof['vectors'] = tfidf.transform(df_standards_prof['full_processed_text'].apply(lambda x: str(x))).todense().tolist()
samples_df['vectors'] = tfidf.transform(samples_df['processed_text'].apply(lambda x: ''.join(x))).todense().tolist()
df_vacancies['vectors'] = tfidf.transform(df_vacancies['processed_text'].apply(lambda x: ''.join(x))).todense().tolist()


# In[162]:


print(full_text['vectors'].shape)
print(df_standards_prof['vectors'].shape)
print(df_vacancies['vectors'].shape)


# In[163]:


print(len(full_text['vectors'][0]))
print(len(df_standards_prof['vectors'][0]))
print(len(df_vacancies['vectors'][0]))


# In[164]:


def most_similar(infer_vector, vectorized_corpus, topn = 5): #vector для сравнения, корпус для сравнения
    df_sim = vectorized_corpus
    vec = np.reshape(infer_vector,(1,-1))
    df_sim['sc'] = vectorized_corpus['vectors'].apply(lambda v : cosine_similarity(vec, np.reshape(v,(1,-1)))[0,0])
    df_sim = df_sim.sort_values(by='sc', ascending=False).head(n = topn)
    return df_sim


# In[187]:


vac_sample = df_vacancies[df_vacancies['labels']=='10'].sample() #

infer_vector = vac_sample.iloc[0]['vectors'] #нужные вектора

ds = df_standards_prof[df_standards_prof['type'].isin(['activities', 'skills','knowledge'])]


# In[188]:


print(vac_sample.index[0])
print(vac_sample.iloc[0]['name'])
print(vac_sample.iloc[0]['type'])
print(vac_sample.iloc[0]['text'])


# In[189]:


cosine_similarity(np.reshape(full_text.vectors[1],(1,-1)),np.reshape(full_text.vectors[0],(1,-1)))


# In[190]:


similar_documents = most_similar(infer_vector, ds, topn = 5)


# In[191]:


similar_documents


# In[197]:


for idx, row in similar_documents.iterrows():
    print('kod_standard: ', row['kod_standard'],'type: ', row['type'],'  score = ',row['sc'])
    print(row['full_text'])
    print('================================')


# In[193]:


print(vac_sample.index[0])
print(vac_sample.iloc[0]['name'])
print(vac_sample.iloc[0]['type'])
print(vac_sample.iloc[0]['text'])


# In[9]:


df_vacancies.head()


# In[88]:


df_vacancies['processed_text'].apply(lambda x: ''.join(x)).head()


# In[86]:


full_text = full_text.drop(columns = ['strvec'])


# In[95]:


np.reshape(infer_vector,(1,-1))


# In[129]:


import codecs
import os
import pymorphy2
from string import ascii_lowercase, digits, whitespace

morph = pymorphy2.MorphAnalyzer()

cyrillic = u"абвгдеёжзийклмнопрстуфхцчшщъыьэюя"

allowed_characters = ascii_lowercase + digits + cyrillic + whitespace

def complex_preprocess(text, additional_allowed_characters = "+#"):
    return ''.join([character if character in set(allowed_characters+additional_allowed_characters) else ' ' for character in text.lower()]).split()

def lemmatize(tokens, filter_pos):
    '''Produce normal forms for russion words using pymorphy2
    '''
    lemmas = []
    tagged_lemmas = []
    for token in tokens:
        parsed_token = morph.parse(token)[0]
        norm = parsed_token.normal_form
        pos = parsed_token.tag.POS        
        if pos is not None:
            if pos not in filter_pos:
                lemmas.append(norm)
                tagged_lemmas.append(norm + "_" + pos)
        else:
            lemmas.append(token)
            tagged_lemmas.append(token+"_")

    return lemmas, tagged_lemmas

def process_text(full_text, filter_pos=("PREP", "NPRO", "CONJ")):
    '''Process a single text and return a processed version
    '''
    single_line_text = full_text.replace('\n',' ')
    preprocessed_text = complex_preprocess(single_line_text)
    lemmatized_text, lemmatized_text_pos_tags = lemmatize(preprocessed_text, filter_pos=filter_pos)

    return { "full_text" : full_text,
    "single_line_text": single_line_text,
    "preprocessed_text": preprocessed_text,
    "lemmatized_text": lemmatized_text,
    "lemmatized_text_pos_tags": lemmatized_text_pos_tags}


# In[130]:


df_standards_prof['full_text'] = df_standards_prof['name_gen_func'] + ' ' + df_standards_prof['name_sim_func'] + ' ' + df_standards_prof['text']


# In[198]:


samples_df


# In[223]:


infer_vector_sample = samples_df.iloc[0] #нужные вектора
infer_vector = infer_vector_sample['vectors']


# In[224]:


similar_documents = most_similar(infer_vector, ds, topn = 5)


# In[225]:


print(infer_vector_sample['text']+'\n')
for idx, row in similar_documents.iterrows():
    print('kod_standard: ', row['kod_standard'],'type: ', row['type'],'  score = ',row['sc'])
    print(row['full_text'])
    print('================================')


# In[226]:


df_most_similar = pd.concat([df_most_similar,similar_documents])


# In[395]:


def most_similar(infer_vector, vectorized_corpus, own = False, topn = 5): #vector для сравнения, корпус для сравнения
    if own == False:
        df_sim = vectorized_corpus
    else:
        df_sim = vectorized_corpus[vectorized_corpus['kod_standard'] == own]
    vec = np.reshape(infer_vector,(1,-1))
    df_sim['sc'] = df_sim['vectors'].apply(lambda v : cosine_similarity(vec, np.reshape(v,(1,-1)))[0,0])
    df_sim = df_sim.dropna()
    df_sim = df_sim.sort_values(by='sc', ascending=False).head(n = topn)
    return df_sim


# In[382]:


df_standards_prof[df_standards_prof['kod_standard'] == '6.001']


# In[400]:


def most_similar(infer_vector, vectorized_corpus, own, topn = 5): #vector для сравнения, корпус для сравнения
    if float(own) !=1:
        df_sim = vectorized_corpus[vectorized_corpus['kod_standard'] == float(own)]
    else:
        df_sim = vectorized_corpus
    vec = np.reshape(infer_vector,(1,-1))
    df_sim['sc'] = df_sim['vectors'].apply(lambda v : cosine_similarity(vec, np.reshape(v,(1,-1)))[0,0])
    df_sim = df_sim.dropna()
    df_sim = df_sim.sort_values(by='sc', ascending=False).head(n = topn)
    return df_sim


# In[401]:


def similarity(vacancies, standards, own = 1):
    df_result = pd.DataFrame(columns = ['vac_text','prof_text','prof_type','prof_code','sc','vector_sim'])
    for index, sample in vacancies.iterrows():
        if own != 1:
            own = sample['kod_standard']
        similar_docs = most_similar(sample['vectors'],standards, own)[['full_text','type','kod_standard','sc']]
        similar_docs['vac_text'] = sample['text']
        similar_docs['name'] = sample['name']
        similar_docs['idx_vacancies'] = sample['Unnamed: 0']
        similar_docs['vector_sim'] = 'TfIdf'
        similar_docs['labels'] = sample['labels']
        similar_docs = similar_docs.rename(columns={
                 'full_text' : 'prof_text',
                 'type' : 'prof_type',
                
                 'kod_standard' : 'prof_code'
                 })   
        similar_docs['prof_name'] = sample['prof_name']

        df_result = pd.concat([df_result,similar_docs])
        print(index)
    return df_result


# In[396]:


similar_samples = similarity(samples_df, df_standards_prof)
#similar_samples.to_csv('similar_samples.csv')


# In[399]:


similar_samples.to_csv('similar_samples.csv')


# In[406]:


similar_samples_own = similarity(samples_df,df_standards_prof, own = 0.1) 
#similar_samples_own.to_csv('similar_samples_own.csv')


# In[407]:


similar_samples_own.to_csv('similar_samples_own.csv')


# In[409]:


similar_samples_own


# In[269]:


similar_samples.to_csv('similar_samples.csv')


# In[288]:


similar_samples = pd.read_csv('similar_samples.csv')


# In[391]:


similar_samples_own.columns


# In[392]:


samples_df


# In[299]:


df_standards_prof


# In[350]:


samples_df['prof_name'] = samples_df['labels'].apply(lambda label: 
                                                     'Специалист по интеграции прикладных решений' if label == 21 else
                                                     'Инженер-проектировщик в области связи (телекоммуникаций)' if label == 20 else
                                                     'Разработчик Web и мультимедийных приложений' if label == 19 else
                                                     'Специалист по тестированию в области информационных технологий' if label == 18 else
                                                     'Архитектор программного обеспечения' if label == 17 else
                                                     'Специалист по информационным ресурсам' if label == 16 else
                                                     'Инженер-радиоэлектронщик' if label == 15 else
                                                     'Программист' if label == 14 else
                                                     'Менеджер по продажам информационно-коммуникационных систем' if label == 13 else
                                                     'Системный программист' if label == 12 else
                                                     'Специалист по администрированию сетевых устройств информационно-коммуникационных систем' if label == 11 else
                                                     'Системный администратор информационно-коммуникационных систем' if label == 10 else
                                                     'Специалист по дизайну графических и пользовательских интерфейсов' if label == 9 else
                                                     'Системный аналитик' if label == 8 else
                                                     'Технический писатель (специалист по технической документации в области информационных технологий)' if label == 7 else
                                                     'Руководитель разработки программного обеспечения' if label == 6 else
                                                     'Руководитель проектов в области информационных технологий' if label == 5 else
                                                     'Менеджер по информационным технологиям' if label == 4 else
                                                     'Менеджер продуктов в области информационных технологий' if label == 3 else
                                                     'Специалист по информационным системам' if label == 2 else
                                                     'Администратор баз данных' if label == 1 else
                                                     'NaN')


# In[353]:


samples_df['kod_standard'] = samples_df['labels'].apply(lambda label: 
                                                        '6.011' if label == 1 else
                                                        '6.015' if label == 2 else
                                                        '6.012' if label == 3 else
                                                        '6.014' if label == 4 else
                                                        '6.016' if label == 5 else
                                                        '6.017' if label == 6 else
                                                        '6.019' if label == 7 else
                                                        '6.022' if label == 8 else
                                                        '6.025' if label == 9 else
                                                        '6.026' if label == 10 else
                                                        '6.027' if label == 11 else
                                                        '6.028' if label == 12 else
                                                        '6.029' if label == 13 else
                                                        '6.001' if label == 14 else
                                                        '6.005' if label == 15 else
                                                        '6.013' if label == 16 else
                                                        '6.003' if label == 17 else
                                                        '6.004' if label == 18 else
                                                        '6.035' if label == 19 else
                                                        '6.007' if label == 20 else
                                                        '6.041' if label == 21 else
                                                        'NaN')

