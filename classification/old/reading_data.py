import pickle
import pandas as pd


def read_all_data():
    dataset_hh = pd.read_csv("../../data/old/old_marked_vacancies_from_hh.csv", header=0)
    dataset_sj = pd.read_csv("../../data/old/old_marked_vacancies_from_sj.csv", header=0)

    return dataset_hh, dataset_sj


def read_tokenized_all_data():

    filename_hh = "../../data/old/old_tokenized_vacancies_from_hh.pkl"
    infile_hh = open(filename_hh, 'rb')
    hh_tokenized = pickle.load(infile_hh)
    infile_hh.close()

    filename_sj = "../../data/old/old_tokenized_vacancies_from_sj.pkl"
    infile_sj = open(filename_sj, 'rb')
    sj_tokenized = pickle.load(infile_sj)
    infile_sj.close()

    return hh_tokenized, sj_tokenized;