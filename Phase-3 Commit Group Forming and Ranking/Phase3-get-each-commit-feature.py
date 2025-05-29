import gc
import json
import os
import re
import logging
import time
from collections import OrderedDict

import nltk.data
from nltk import word_tokenize
from nltk.corpus import stopwords

import numpy as np
import pandas as pd
from tqdm import tqdm
import multiprocessing as mp

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
stemmer = nltk.stem.SnowballStemmer('english')
stop_words = set(stopwords.words('english'))

tqdm.pandas()

def RemoveGit(str):
    gitPattern = '[Gg]it-svn-id'
    return re.sub(gitPattern, ' ', str)


def clean_en_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9 ]', ' ', text) 
    text = ' '.join(text.split())
    return text


def textProcess(text):
    final = []

    text = RemoveGit(text)
    sentences = tokenizer.tokenize(text)
    for sentence in sentences:
        sentence = clean_en_text(sentence)
        word_tokens = word_tokenize(sentence)
        word_tokens = [word for word in word_tokens if word not in stop_words]
        for word in word_tokens:
            final.append(str(stemmer.stem(word)))

    if len(final) == 0:
        text = ' '
    else:
        text = ' '.join(final)
    return text



data_df = pd.read_csv('')


print('Start dataset processing at', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
df_each = pd.read_csv('')
data_df = df

handcrafted_columns = ['cve', 'commit', 'label', 'group_id', 'desc', 'msg_text', 'deepseek_text',
                'addcnt', 'delcnt', 'totalcnt', 'issue_cnt', 'web_cnt', 'bug_cnt', 'cve_cnt',
                'cve_match', 'bug_match', 'issue_match', 'cwe_match',
                'time_dis', 'vuln_type_1', 'vuln_type_2', 'vuln_type_3',
                'filepath_same_cnt', 'filepath_same_ratio', 'filepath_unrelated_cnt',
                'file_same_cnt', 'file_same_ratio', 'file_unrelated_cnt',
                'func_same_cnt', 'func_same_ratio', 'func_unrelated_cnt',
                'inter_token_cwe_cnt', 'inter_token_cwe_ratio', 'commit_vuln_tfidf', 'commit_vuln_ds_tfidf',
                'mess_shared_num', 'mess_shared_ratio', 'mess_max', 'mess_sum', 'mess_mean', 'mess_var',
                'ds_shared_num', 'ds_shared_ratio', 'ds_max', 'ds_sum', 'ds_mean', 'ds_var', 'patch_score']    
data_df = data_df[handcrafted_columns]
data_df.to_csv('')
print('Start dataset processing at', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
