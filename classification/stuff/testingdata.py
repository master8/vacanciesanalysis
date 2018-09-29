import pickle

import pandas as pd


filename_hh = "../data/old/old_tokenized_vacancies_from_hh.pkl"
infile_hh = open(filename_hh, 'rb')
hh_tokenized = pickle.load(infile_hh)
infile_hh.close()

filename_sj = "../data/old/old_tokenized_vacancies_from_sj.pkl"
infile_sj = open(filename_sj, 'rb')
sj_tokenized = pickle.load(infile_sj)
infile_sj.close()