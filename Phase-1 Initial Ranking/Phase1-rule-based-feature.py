import datetime
import os
import gc
import warnings
import json
import glob
import shutil
import pandas as pd
from tqdm import tqdm
import multiprocessing as mp
from collections import Counter
from util import *
import ast
from nltk import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
stop_words = set(stopwords.words('english'))
stemmer = nltk.stem.SnowballStemmer('english')
stop_words.add('.')

tqdm.pandas()

with open("../data/vuln_type_impact.json", 'r') as f:
    vuln_type_impact = json.load(f)
vuln_type = set(vuln_type_impact.keys())
vuln_impact = set()
for value in vuln_type_impact.values():
    vuln_impact.update(value)


def convert_lowercase_remove_symbol(item):
    item = item.lower()
    item = re.sub(r'[^a-zA-Z0-9. ]', ' ', item)  
    item = " ".join(item.split())
    return item


def tokenize_text(text):
    token_list = []
    text = convert_lowercase_remove_symbol(text)
    sentences = tokenizer.tokenize(text)
    for sentence in sentences:
        word_tokens = word_tokenize(sentence)
        word_tokens = [word for word in word_tokens if word not in stop_words]
        for word in word_tokens:
            token_list.append(str(stemmer.stem(word)))

    return token_list


def re_filepath(text):
    res = []
    find = re.findall(
        r'(([a-zA-Z0-9/_-]+)\.(cpp|cc|cxx|cp|CC|hpp|hh|C|c|h|py|php|java|js|rb|go|html|sh|pm|m|phtml|yaml|ctp))(?=\s|\b)',
        text)
    for item in find:
        res.append(item[0])
    return res


def re_file(text):
    res = []
    find = re.findall(
        r'(([a-zA-Z0-9_-]+)\.(cpp|cc|cxx|cp|CC|hpp|hh|C|c|h|py|php|java|js|rb|go|html|sh|pm|m|phtml|yaml|ctp))(?=\s|\b)',
        text)
    for item in find:
        res.append(item[0])
    return res


def re_func(text):
    res = []
    find = re.findall("(([a-zA-Z0-9]+_)+[a-zA-Z0-9]+.{2})", text)  
    for item in find:
        item = item[0]
        if item[-1] == ' ' or item[-2] == ' ':
            res.append(item[:-2])

    find = re.findall("(([a-zA-Z0-9]+_)*[a-zA-Z0-9]+\\(\\))", text)  
    for item in find:
        item = item[0]
        res.append(item[:-2])

    find = re.findall("(([a-zA-Z0-9]+_)*[a-zA-Z0-9]+ function)", text)  
    for item in find:
        item = item[0]
        res.append(item[:-9])

    res = list(set(res))
    macro_list = [item for item in res if item.isupper()]
    return list(set(res) - set(macro_list)), macro_list


def get_tokens(text, List):
    list_src = [item for item in List if item in text]
    return set(list_src)


def get_cwe_token(cwe):
    cwe_token = []
    for each in cwe:
        cwe_id, cwe_text = each[0], each[1]
        if cwe_text == 'NVD-CWE-noinfo':
            continue
        else:
            cwe_token.extend(tokenize_text(cwe_text))

    return cwe_token


def get_url_id(url):
    return re.findall(r'\b\d+\b', url)


def get_urls_id_list(urls):
    bug_id_list = []
    issue_id_list = []
    for url in urls:
        if 'bug' in url or 'Bug' in url or 'bid' in url:
            bug_id_list.extend(get_url_id(url))
        if 'issue' in url or 'Issue' in url:
            issue_id_list.extend(get_url_id(url))

    return list(set(bug_id_list)), list(set(issue_id_list))

 
    list1 = re.findall(r'\bbug[^a-zA-Z0-9,.]{0,3}([0-9]{1,7})[^0-9]', text, re.IGNORECASE)
    list2 = []
    for url in msg_urls:
        if 'bug' in url or 'Bug' in url or 'bid' in url:
            list2.extend(get_url_id(url))
    return set(list1 + list2)


import requests
from bs4 import BeautifulSoup

with open('data/ghsa_dict.pkl', 'rb') as f:
    ghsa_to_cve_dict = pickle.load(f)

def re_cve(item, msg_urls):
    cve_list = re.findall(r'(CVE-[0-9]{4}-[0-9]{1,7})', item, re.IGNORECASE)
    for url in msg_urls:
        ghsa_ids = re.findall(r'GHSA-[a-z0-9]{4}-[a-z0-9]{4}-[a-z0-9]{4}', url)
        for ghsa_id in ghsa_ids:
            if ghsa_id in ghsa_to_cve_dict and ghsa_to_cve_dict[ghsa_id]:
                cve_list.extend(ghsa_to_cve_dict[ghsa_id])
            else:
                try:
                    r = requests.get(f'https://github.com/advisories/{ghsa_id}')
                    r.encoding = r.apparent_encoding
                    demo = r.text
                    soup = BeautifulSoup(demo, 'html.parser')
                    it = iter(soup.find_all('div', 'color-fg-muted'))
                    for tag in it:
                        if tag.string is not None:
                            text = tag.string.strip()
                            if re.match(r'CVE-[0-9]{4}-[0-9]{1,7}', text):
                                cve_list.append(text)
                except Exception as e:
                    with open('data/ghsa_error.txt', 'a') as f:
                        f.write(f'Error processing {ghsa_id}: {e}\n')
    return set(cve_list)


def re_issue(item, msg_urls):
    list1 = re.findall('[iI]ssue[^0-9]{0,5}([0-9]{1,7})[^0-9]', item)
    list2 = []
    for url in msg_urls:
        found = re.findall(r".*/issues/.*?(\d+).*$", url)
        list2.extend(found)
    return set(list1 + list2)


def re_weblink_len(item):
    link_re = r'https?://[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]'
    return len(set(re.findall(link_re, item)))


def calculate_match(commit_cves, cve, commit_bugs, cve_bugs, commit_issues, cve_issues):
    cve_match = 0
    for item in commit_cves:
        if item.lower() in cve.lower():
            cve_match = 1
            break

    issue_match = 0
    for issue in commit_issues:
        if issue in cve_issues:
            issue_match = 1
            break

    bug_match = 0
    for bug in commit_bugs:
        if bug in cve_bugs:
            bug_match = 1
            break

    return cve_match, bug_match, issue_match


def feature_time(committime, cvetime):
    committime = datetime.datetime.strptime(committime, '%Y%m%d')
    cvetime = datetime.datetime.strptime(cvetime, '%Y%m%d')
    time_dis = abs((cvetime - committime).days)
    return time_dis


def get_commit_info(mess, diff_code, cve_id, links, cve_time, commit_time, msg_urls, cwe, cve_bug_id, cve_issue_id):

    bugs = re_bug(mess, msg_urls)
    bugs_cnt = len(bugs)

    cves = re_cve(mess, msg_urls)
    cves_cnt = len(cves)

    issues = re_issue(mess, msg_urls)
    issues_cnt = len(issues)

    cve_match, bug_match, issue_match = calculate_match(cves, cve_id, bugs, cve_bug_id, issues, cve_issue_id)

    links_cnt = re_weblink_len(mess)

    mess_filtered = convert_lowercase_remove_symbol(mess)

    cwe_match = 0
    for cwe_id, cwe_text in cwe:
        cwe_id = convert_lowercase_remove_symbol(cwe_id)
        if cwe_id in mess_filtered:
            cwe_match = 1
            break

        if '(' in cwe_text:
            cwe_list = cwe_text.split('(')
            if convert_lowercase_remove_symbol(cwe_list[0]) in mess_filtered or \
                    convert_lowercase_remove_symbol(cwe_list[1]) in mess_filtered:
                cwe_match = 1
                break
        elif convert_lowercase_remove_symbol(cwe_text) in mess_filtered:
            cwe_match = 1
            break

    type_set = set()
    for value in vuln_type:
        if value in mess_filtered:
            type_set.add(value)

    impact_set = set()
    for value in vuln_impact:
        if value in mess_filtered:
            impact_set.add(value)

    mess_token = tokenize_text(mess_filtered)

    has_fix = 0
    if 'fix' in mess_token: 
        has_fix = 1

    diff_code_lines = diff_code.split('\n')

    addcnt, delcnt = 0, 0
    filepaths, files, funcs, code_token = [], [], [], []
    for line in diff_code_lines:
        if re.match(r'^diff\s+--git', line):
            filepath = line.split(' ')[-1].strip()[2:]
            filepaths.append(filepath)
            file = filepath.split('/')[-1]
            files.append(file)
        elif line.startswith('@@ '):
            funcname = line.split('@@')[-1].strip()
            funcname = funcs_preprocess(funcname)
            funcs.append(funcname)
        else:
            if line.startswith('+') and not line.startswith('++'):  
                addcnt += 1
                line_filtered = convert_lowercase_remove_symbol(line)
                code_token.extend(tokenize_text(line_filtered))
            elif line.startswith('-') and not line.startswith('--'):  
                delcnt += 1
                line_filtered = convert_lowercase_remove_symbol(line)
                code_token.extend(tokenize_text(line_filtered))

    time_dis = feature_time(cve_time, commit_time)
    funcs = [func for func in funcs if func]

    return cves, bugs, type_set, impact_set, bugs_cnt, cves_cnt, issues_cnt, links_cnt, cve_match, bug_match, \
        issue_match, cwe_match, has_fix, addcnt, delcnt, addcnt + delcnt, set(filepaths), set(files), set(funcs), \
        mess_token, code_token, time_dis


def multi_get_commit_info(row):
    return get_commit_info(row['msg_text'], row['diff_code'], row['cve'], row['cve_links'], row['cve_time'],
                           row['commit_time'], row['msg_url'], row['cwe'], row['cve_bug_id'], row['cve_issue_id'])


def get_vuln_loc(cve_items, commit_items):
    same_cnt = 0
    commit_items = list(set(commit_items))
    for commit_item in commit_items:
        for cve_item in cve_items:
            if cve_item == commit_item:  
                same_cnt += 1
                break
    same_ratio = same_cnt / (len(commit_items) + 1)
    unrelated_cnt = len(cve_items) - same_cnt
    return same_cnt, same_ratio, unrelated_cnt


def multi_get_vuln_loc_filepath(row):
    return get_vuln_loc(row['cve_filepaths'], row['mess_filepaths'])


def multi_get_vuln_loc_file(row):
    return get_vuln_loc(row['cve_files'], row['mess_files'])


def multi_get_vuln_loc_func(row):
    return get_vuln_loc(row['cve_funcs'], row['mess_funcs'])


def get_vuln_type_related(vuln_type, vuln_impact, commit_type, commit_impact, vuln_type_impact):
    l1, l2, l3 = 0, 0, 0  

    for nvd_item in vuln_type:
        for commit_item in commit_type:
            if nvd_item == commit_item:
                l1 += 1
            else:
                l3 += 1

    for nvd_item in vuln_type:
        for commit_item in commit_impact:
            if commit_item in vuln_type_impact.get(nvd_item):
                l2 += 1
            else:
                l3 += 1
                
    for commit_item in commit_type:
        for nvd_item in vuln_impact:
            if nvd_item in vuln_type_impact.get(commit_item):
                l2 += 1
            else:
                l3 += 1

    cnt = l1 + l2 + l3 + 1

    return l1 / cnt, l2 / cnt, (l3 + 1) / cnt


def multi_get_vuln_type_related(row):
    return get_vuln_type_related(row['vuln_type'], row['vuln_impact'], row['mess_type'], row['mess_impact'],
                                 vuln_type_impact)


def vuln_commit_token(cwe_tokens, commit_tokens):
    commit_tokens_set = set(commit_tokens)
    cwe_tokens_set = set(cwe_tokens)

    inter_token_cwe = inter_token(cwe_tokens_set, commit_tokens_set)
    inter_token_cwe_cnt = len(inter_token_cwe)
    inter_token_cwe_ratio = inter_token_cwe_cnt / (1 + len(cwe_tokens_set))

    return inter_token_cwe_cnt, inter_token_cwe_ratio


def get_shared_token_num_ratio(desc_cve_token, mess_or_code_token):
    c1 = Counter(desc_cve_token)
    c2 = Counter(mess_or_code_token)
    c3 = c1 & c2  

    shared_num = len(c3.keys())
    shared_ratio = shared_num / (len(c1.keys()) + 1)
    c3_value = list(c3.values())
    if len(c3_value) == 0:
        c3_value = [0]
    return shared_num, shared_ratio, max(c3_value), sum(c3_value), np.mean(c3_value), np.var(c3_value)


def multi_vuln_commit_token(row):
    return vuln_commit_token(row['cwe_token'], row['commit_token'])


def multi_get_vuln_desc_text1(row):
    return get_shared_token_num_ratio(row['cve_token'], row['mess_token'])


def multi_get_vuln_desc_text2(row):
    return get_shared_token_num_ratio(row['cve_token'], row['code_token'])


def compute_similarity(args):
    group, cve = args

    group['commit_text'] = group['commit_token'].apply(lambda items: ' '.join([item for item in items]))
    group['cve_text'] = group['cve_token'].apply(lambda items: ' '.join([item for item in items]))

    vectorizer = TfidfVectorizer()
    try:
        vectorizer.fit(group['commit_text'])
    except Exception as e:
        group['commit_text'].fillna('', inplace=True)
        vectorizer.fit(group['commit_text'])

    similarity_scores = []

    for _, row in group.iterrows():
        desc_tfidf = vectorizer.transform([row['cve_text']])
        mess_diff_code_tfidf = vectorizer.transform([row['commit_text']])
        similarity = cosine_similarity(desc_tfidf, mess_diff_code_tfidf).diagonal()[0]
        similarity_scores.append(similarity)

    similarity_data = pd.DataFrame()
    similarity_data['cve'] = group['cve']
    similarity_data['commit'] = group['commit']
    similarity_data['commit_vuln_tfidf'] = similarity_scores

    return similarity_data

def safe_literal_eval(val):
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError):
        return val


files = glob.glob("")
dataset_names = {}
df_list = [pd.read_csv(file).fillna('') for file in files]

output_path = ''
if not os.path.exists(output_path):
    os.makedirs(output_path)

for index, df_one in enumerate(df_list):
    df_name = dataset_names[index]
    df_vuln = pd.read_csv('')

    print('Start', df_name, 'dataset processing at', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    df_one['diff_code'] = df_one['diff_code'].fillna('')
    df_one['commit_time'] = pd.to_datetime(df_one['commit_time'])
    df_one['commit_time'] = df_one['commit_time'].dt.strftime('%Y%m%d')
    df_one['commit_time'] = df_one['commit_time'].astype(str)

    dirpath = '' + df_name  
    if os.path.exists(dirpath):
        shutil.rmtree(dirpath)
    os.makedirs(dirpath)

    total_cve = df_one.cve.unique()
    each_epoch_cve_num = 40
    epoch = int((len(total_cve) + each_epoch_cve_num - 1) / each_epoch_cve_num)
    num_cores = 40
    print('----- separate dataset into {} epochs'.format(epoch))

    for i in tqdm(range(epoch)):

        cve_list = total_cve[i * each_epoch_cve_num:(i + 1) * each_epoch_cve_num]
        df = df_one[df_one['cve'].isin(cve_list)]

        df = df.merge(df_vuln, how='left', on='cve')
        df['msg_url'] = df['msg_url'].apply(safe_literal_eval)
        df['msg_sign'] = df['msg_sign'].apply(safe_literal_eval)

        print('Start', df_name, 'dataset\'s features generating at',
              time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

        with mp.Pool(num_cores) as pool:
            results = list(tqdm(pool.imap(multi_get_commit_info, [row for _, row in df.iterrows()]), total=len(df),
                                desc='commit feature 1'))
        df['mess_cves'], df['mess_bugs'], df['mess_type'], df['mess_impact'], df['bug_cnt'], df['cve_cnt'], df[
            'issue_cnt'], df['web_cnt'], df['cve_match'], df['bug_match'], df['issue_match'], df['cwe_match'], df[
            'has_fix'], df['addcnt'], df['delcnt'], df['totalcnt'], df['mess_filepaths'], df['mess_files'], df[
            'mess_funcs'], df['mess_token'], df['code_token'], df['time_dis'] = zip(*results)

        df['commit_token'] = df['mess_token'] + df['code_token']

        with mp.Pool(num_cores) as pool:
            results = list(
                tqdm(pool.imap(multi_get_vuln_type_related, [row for _, row in df.iterrows()]), total=len(df),
                     desc='vuln_commit type and impact feature'))
        df['vuln_type_1'], df['vuln_type_2'], df['vuln_type_3'] = zip(*results)

        with mp.Pool(num_cores) as pool:
            results = list(
                tqdm(pool.imap(multi_get_vuln_loc_filepath, [row for _, row in df.iterrows()]), total=len(df),
                     desc='filepath feature'))
        df['filepath_same_cnt'], df['filepath_same_ratio'], df['filepath_unrelated_cnt'] = zip(*results)

        with mp.Pool(num_cores) as pool:
            results = list(
                tqdm(pool.imap(multi_get_vuln_loc_file, [row for _, row in df.iterrows()]), total=len(df),
                     desc='file feature'))
        df['file_same_cnt'], df['file_same_ratio'], df['file_unrelated_cnt'] = zip(*results)

        with mp.Pool(num_cores) as pool:
            results = list(
                tqdm(pool.imap(multi_get_vuln_loc_func, [row for _, row in df.iterrows()]), total=len(df),
                     desc='func feature'))
        df['func_same_cnt'], df['func_same_ratio'], df['func_unrelated_cnt'] = zip(*results)

        with mp.Pool(num_cores) as pool:
            results = list(tqdm(pool.imap(multi_vuln_commit_token, [row for _, row in df.iterrows()]), total=len(df),
                                desc='vuln-cwe_commit shared words'))
        df['inter_token_cwe_cnt'], df["inter_token_cwe_ratio"] = zip(*results)

        with mp.Pool(num_cores) as pool:
            results = list(tqdm(pool.imap(multi_get_vuln_desc_text1, [row for _, row in df.iterrows()]), total=len(df),
                                desc='vuln_mess shared words'))
        df['mess_shared_num'], df['mess_shared_ratio'], df['mess_max'], df['mess_sum'], df['mess_mean'], df[
            'mess_var'] = zip(*results)

        with mp.Pool(num_cores) as pool:
            results = list(tqdm(pool.imap(multi_get_vuln_desc_text2, [row for _, row in df.iterrows()]), total=len(df),
                                desc='vuln_diff-code shared words'))
        df['code_shared_num'], df['code_shared_ratio'], df['code_max'], df['code_sum'], df['code_mean'], df[
            'code_var'] = zip(*results)

        single_cve_group = df.groupby('cve')
        with mp.Pool(num_cores) as pool:
            result_list = list(tqdm(pool.imap(compute_similarity, [(group, cve) for cve, group in single_cve_group]),
                                    total=len(cve_list), desc='Computing similarity scores'))
        tfidf_df = pd.concat(result_list, ignore_index=True)
        df = df.merge(tfidf_df, how='left', on=['cve', 'commit'])

        df = df.drop(
            ['cve_token', 'cwe', 'cwe_token', 'mess_token', 'code_token', 'commit_token', 'diff_code', 'msg_sign',
             'msg_url', 'msg_text', 'author', 'committer', 'links', 'desc', 'cve_macro', 'desc_filtered',
             'vuln_type', 'vuln_impact', 'cve_filepaths', 'cve_files', 'cve_funcs', 'cve_links'], axis=1)
        df.to_csv(''.format(i), index=False)

    gc.collect()

    files = glob.glob(dirpath + '/*.csv')
    m = {}
    for file in files:
        idx = int(re.search('([0-9]+).csv', file).group(1))
        m[idx] = file

    l = []
    for i in range(epoch):
        tmp = pd.read_csv(m[i])
        l.append(tmp)

    data_df = pd.concat(l)
    data_df = data_df.reset_index(drop=True)

    df_saved = data_df[['cve', 'commit', 'mess_cves', 'mess_bugs', 'mess_type', 'mess_impact',
                        'mess_filepaths', 'mess_files', 'mess_funcs']]
    df_saved.to_csv(output_path + df_name + '_commit_info.csv', index=False)
    data_df = data_df.drop(['mess_cves', 'mess_bugs', 'mess_type', 'mess_impact'], axis=1)
    data_df.to_csv(output_path + df_name + '_feature.csv', index=False)




