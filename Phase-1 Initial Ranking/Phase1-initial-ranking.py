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

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer
from transformers import (AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaTokenizer, RobertaModel, AutoModel)
from transformers import AutoModelForSeq2SeqLM, AdamW, get_linear_schedule_with_warmup

from sklearn.model_selection import train_test_split


os.environ['CUDA_VISIBLE_DEVICES'] = '6,7'
device = torch.device("cuda")
n_gpu = 2
num_epochs = 20
MULTI_GPU = 1

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
stemmer = nltk.stem.SnowballStemmer('english')
stop_words = set(stopwords.words('english'))

tqdm.pandas()

def get_rank(df1, sortby, ascending=False):
    gb = df1.groupby('cve')
    l = []
    for item1, item2 in gb:
        item2 = item2.reset_index()
        item2 = item2.sort_values(sortby + ['commit'], ascending=ascending)
        item2 = item2.reset_index(drop=True).reset_index()
        l.append(item2[['index', 'level_0']])

    df1 = pd.concat(l)
    df1['rank'] = df1['level_0'] + 1
    df1 = df1.sort_values(['index'], ascending=True).reset_index(drop=True)
    return df1['rank']


def get_metrics_N(test, rankname='rank', N=10):
    cve_list = []
    cnt = 0
    total = []
    gb = test.groupby('cve')
    for item1, item2 in gb:
        item2 = item2.sort_values([rankname], ascending=True).reset_index(drop=True)
        idx = item2[item2.label == 1].index[-1] + 1
        if idx <= N:
            total.append(idx)
            cnt += 1
        else:
            total.append(N)
            cve_list.append(item1)
    return cnt / len(total), np.mean(total)


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


def diffcodeProcess(diffcode):
    added_code = ''
    deleted_code = ''
    added_annotation = ''
    deleted_annotation = ''
    lines = diffcode.split('\n')
    for line in lines:
        if line.startswith('+') and not line.startswith('++'):
            line = line[1:].strip()
            added_code = added_code + line + ' '
        if line.startswith('-') and not line.startswith('--'):
            line = line[1:].strip()
            deleted_code = deleted_code + line + ' '

    return added_code, deleted_code, added_annotation, deleted_annotation


def convert_examples_to_features(desc, mess, tokenizer, max_seq_length):
    desc_token = tokenizer.tokenize(desc)
    mess_token = tokenizer.tokenize(mess)

    if len(desc_token) + len(mess_token) > max_seq_length - 3:
        if len(desc_token) > (max_seq_length - 3) / 2 and len(mess_token) > (max_seq_length - 3) / 2:
            desc_token = desc_token[:int((max_seq_length - 3) / 2)]
            mess_token = mess_token[:max_seq_length - 3 - len(desc_token)]
        elif len(desc_token) > (max_seq_length - 3) / 2:
            desc_token = desc_token[:max_seq_length - 3 - len(mess_token)]
        elif len(mess_token) > (max_seq_length - 3) / 2:
            mess_token = mess_token[:max_seq_length - 3 - len(desc_token)]
    combined_token = [tokenizer.cls_token] + desc_token + [tokenizer.sep_token] + mess_token + [tokenizer.sep_token]
    input_ids_text = tokenizer.convert_tokens_to_ids(combined_token)
    if len(input_ids_text) < max_seq_length:
        padding_length = max_seq_length - len(input_ids_text)
        input_ids_text += [tokenizer.pad_token_id] * padding_length
    input_ids_text = torch.tensor(input_ids_text)
    assert len(input_ids_text) == max_seq_length, 'Length of input_ids_text is error!'

    attention_mask_text = input_ids_text.ne(tokenizer.pad_token_id).to(torch.int64)
    return input_ids_text, attention_mask_text


class NewDataset(Dataset):
    def __init__(self, cve_list, ns=0, ns_num=5):
        data_df = pd.read_csv('../data/single-patch_dataset.csv')
        data_df = data_df[data_df['cve'].isin(cve_list)]
        df_1 = pd.read_csv('../data/multi-patch_dataset.csv')
        df_1 = df_1[df_1['cve'].isin(cve_list)]
        data_df = pd.concat([data_df, df_1])
        data_df = data_df.reset_index(drop=True)

        if ns != 0:
            positive_samples = data_df[data_df['label'] == 1]
            negative_samples = data_df[data_df['label'] == 0]

            balanced_negative_samples = pd.DataFrame() 
            for cve_id, group in negative_samples.groupby('cve'):
                if len(group) > ns_num:
                    sampled_group = group.sample(n=ns_num, random_state=42)
                else:
                    sampled_group = group
                balanced_negative_samples = pd.concat([balanced_negative_samples, sampled_group])

            data_df = pd.concat([positive_samples, balanced_negative_samples])
            data_df = data_df.reset_index(drop=True)

        df_each = pd.read_csv('')
        df = data_df.merge(df_each[['cve', 'commit', 'label', 'msg_text', 'diff_code']], how='inner', on=['cve', 'commit', 'label'])
        df = df.reset_index(drop=True)
        for i in range(1, 8):
            df_each = pd.read_csv('')
            merge_df = data_df.merge(df_each[['cve', 'commit', 'label', 'msg_text', 'diff_code']], how='inner', on=['cve', 'commit', 'label'])
            df = pd.concat([df, merge_df], ignore_index=True)
        for i in range(4):       
            df_each = pd.read_csv('')
            merge_df = data_df.merge(df_each[['cve', 'commit', 'label', 'msg_text', 'diff_code']], how='inner', on=['cve', 'commit', 'label'])
            df = pd.concat([df, merge_df], ignore_index=True)
        data_df = df

        df_each = pd.read_csv('')
        df = data_df.merge(df_each, how='inner', on=['cve', 'commit', 'label'])
        df = df.reset_index(drop=True)
        for i in range(1, 8):
            df_each = pd.read_csv('')
            merge_df = data_df.merge(df_each, how='inner', on=['cve', 'commit', 'label'])
            df = pd.concat([df, merge_df], ignore_index=True)
        for i in range(4):        
            df_each = pd.read_csv('')
            merge_df = data_df.merge(df_each, how='inner', on=['cve', 'commit', 'label'])
            df = pd.concat([df, merge_df], ignore_index=True)

        df_cve = pd.read_csv('../data/CVE_info.csv')
        df_cve = df_cve[['cve', 'desc']]
        df = df.merge(df_cve, how='left', on='cve')
        df = df.fillna('')

        with mp.Pool(mp.cpu_count()) as pool:
            df['desc'] = list(tqdm(pool.imap(textProcess, df['desc']), total=len(df),
                                                 desc='Processing cve_desc'))
        self.desc = df['desc']
        with mp.Pool(mp.cpu_count()) as pool:
            df['msg_text'] = list(tqdm(pool.imap(textProcess, df['msg_text']), total=len(df),
                                                 desc='Processing msg_text'))
        self.mess = df['msg_text']
        with mp.Pool(mp.cpu_count()) as pool:
            results = list(tqdm(pool.imap(diffcodeProcess, df['diff_code']), total=len(df),
                                             desc='Processing diff_code'))
        df['added_code'], df['deleted_code'], df['added_an'], df['deleted_an'] = zip(*results)

        self.added_code = df['added_code']
        self.deleted_code = df['deleted_code']
        self.cve = df['cve']
        self.commit = df['commit']
        self.label = df['label']


        handcrafted_columns = ['addcnt', 'delcnt', 'totalcnt', 'issue_cnt', 'bug_cnt', 'cve_cnt',
                        'cve_match', 'bug_match', 'issue_match', 'cwe_match',
                        'time_dis', 'vuln_type_1', 'vuln_type_2', 'vuln_type_3',
                        'filepath_same_cnt', 'filepath_same_ratio', 'filepath_unrelated_cnt',
                        'file_same_cnt', 'file_same_ratio', 'file_unrelated_cnt',
                        'func_same_cnt', 'func_same_ratio', 'func_unrelated_cnt',
                        'inter_token_cwe_cnt', 'inter_token_cwe_ratio', 'commit_vuln_tfidf',
                        'mess_shared_num', 'mess_shared_ratio', 'mess_max', 'mess_sum', 'mess_mean', 'mess_var',
                        'code_shared_num', 'code_shared_ratio', 'code_max', 'code_sum', 'code_mean', 'code_var']    
        handcrafted_feature = df[handcrafted_columns]
        self.handcrafted = handcrafted_feature.to_numpy() 

        self.text_tokenizer = RobertaTokenizer.from_pretrained('../pretrained_model/roberta-large')
        self.code_tokenizer = AutoTokenizer.from_pretrained('../pretrained_model/codereviewer')

        gc.collect()

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        max_seq_length = 512
        desc = self.desc[index] if isinstance(self.desc[index], str) else ''
        mess = self.mess[index] if isinstance(self.mess[index], str) else ''
        input_ids_text, attention_mask_text = convert_examples_to_features(desc, mess, self.text_tokenizer,
                                                                           max_seq_length)

        added_code = self.added_code[index] if isinstance(self.added_code[index], str) else ''
        deleted_code = self.deleted_code[index] if isinstance(self.deleted_code[index], str) else ''
        input_ids_diff, attention_mask_diff = convert_examples_to_features(added_code, deleted_code,
                                                                           self.code_tokenizer, max_seq_length)

        sample = (input_ids_text, attention_mask_text, input_ids_diff, attention_mask_diff,
                  torch.tensor(self.handcrafted[index]), torch.tensor(self.label[index]),
                  self.cve[index], self.commit[index])

        return sample


class NewModel(nn.Module):
    def __init__(self):
        super(NewModel, self).__init__()

        self.hc_dim = 38
        self.s_dim = 32
        config = RobertaConfig.from_pretrained('../pretrained_model/roberta-large')
        config.num_labels = 1
        self.textEncoder = AutoModel.from_pretrained('../pretrained_model/roberta-large', config=config)
        self.codeEncoder = AutoModelForSeq2SeqLM.from_pretrained('../pretrained_model/codereviewer').encoder

        self.fc1 = nn.Linear(self.textEncoder.config.hidden_size, self.s_dim)
        self.fc2 = nn.Linear(self.codeEncoder.config.hidden_size, self.s_dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.s_dim * 2 + self.hc_dim, (self.s_dim * 2 + self.hc_dim) // 2),
            nn.ReLU(),
            nn.Linear((self.s_dim * 2 + self.hc_dim) // 2, 2)
        )

        self.criterion = nn.CrossEntropyLoss()

        for param in self.textEncoder.parameters():
            param.requires_grad = True
        for param in self.codeEncoder.parameters():
            param.requires_grad = True

    def forward(self, input_ids_text, attention_mask_text, input_ids_diff, attention_mask_diff, handcrafted, label=None):
        text_output = self.textEncoder(input_ids=input_ids_text, attention_mask=attention_mask_text)[1] 
        code_output = self.codeEncoder(input_ids=input_ids_diff, attention_mask=attention_mask_diff).last_hidden_state[:, 0, :]

        text_output = self.fc1(text_output)     
        combine_output = torch.cat([text_output, code_output], dim=-1)    
        combine_output = torch.cat([combine_output, handcrafted], dim=-1)  

        logits = self.mlp(combine_output)  

        prob = torch.softmax(logits, -1)
        if label is not None:
            loss = self.criterion(logits, label)
            return loss, prob
        else:
            return prob

def train(model, train_dataloader, result_dir, max_grad_norm=0.1):

    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters()], 'weight_decay': 0.0}]
    optimizer = AdamW(optimizer_grouped_parameters, lr=1e-4)


    max_steps = len(train_dataloader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=max_steps * 0.1,
                                                num_training_steps=max_steps)


    torch.set_grad_enabled(True)
    model.zero_grad()
    model.train()
    for idx in range(num_epochs):
        bar = tqdm(train_dataloader, total=len(train_dataloader))
        losses = []
        for step, batch in enumerate(bar):
            input_ids_text = batch[0].to(device)
            attention_mask_text = batch[1].to(device)
            input_ids_diff = batch[2].to(device)
            attention_mask_diff = batch[3].to(device)
            handcrafted = batch[4].float().to(device)
            label = batch[5].long().to(device)

            loss, prob = model(input_ids_text, attention_mask_text, input_ids_diff, attention_mask_diff, handcrafted,
                                  label)

            if n_gpu > 1:
                loss = loss.mean()  
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            losses.append(loss.item())
         
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        print("epoch {} loss {}".format(idx, round(float(np.mean(losses)), 3)))

        if (idx + 1) % 5 == 0:
            model_file = result_dir + '/checkpoint_Phase1_model.bin'
            model_to_save = model.module if hasattr(model, 'module') else model
            torch.save(model_to_save.state_dict(), model_file)

            full_model_file = result_dir + '/checkpoint_Phase1_full_model.bin'
            torch.save(model, full_model_file)

        torch.cuda.empty_cache()

def test(model, test_dataloader, result_dir, info):
    prob_list = []
    label_list = []
    cve_list = []
    commit_list = []

    model.eval()
    bar = tqdm(test_dataloader, total=len(test_dataloader))
    for step, batch in enumerate(bar):
        input_ids_text = batch[0].to(device)
        attention_mask_text = batch[1].to(device)
        input_ids_diff = batch[2].to(device)
        attention_mask_diff = batch[3].to(device)
        handcrafted = batch[4].float().to(device)
        label = batch[5].long().to(device)
        cve = batch[6]
        commit = batch[7]

        with torch.no_grad():
            prob = model(input_ids_text, attention_mask_text, input_ids_diff, attention_mask_diff, handcrafted)

            prob_list.append(prob.cpu().numpy())
            label_list.append(label.cpu().numpy())
            cve_list.append(list(cve))
            commit_list.append(list(commit))

    torch.cuda.empty_cache()
    cve_list = np.concatenate(cve_list, 0)
    prob_list = np.concatenate(prob_list, 0)
    prob_list = prob_list[:, 1]
    label_list = np.concatenate(label_list, 0)
    commit_list = np.concatenate(commit_list, 0)

    p_data = {
        'cve': cve_list,
        'commit': commit_list,
        'predict': prob_list,
        'label': label_list
    }

    result_csv = pd.DataFrame(p_data)
    result_csv.to_csv(result_dir + '/rank_result_origin_' + info + '.csv', index=False)
    result_csv['rank'] = get_rank(result_csv, ['predict'], ascending=False)
    result_csv.to_csv(result_dir + '/rank_result_' + info + '.csv', index=False)

    for i in range(10):
        recall, manual_efforts = get_metrics_N(result_csv, rankname='rank', N=i+1)
        print('Top-' + str(i + 1), 'recall: ', recall, 'manual_efforts: ', manual_efforts)
    recall, manual_efforts = get_metrics_N(result_csv, rankname='rank', N=100)
    print('Top-100', 'recall: ', recall, 'manual_efforts: ', manual_efforts)

    recall, manual_efforts = get_metrics_N(result_csv, rankname='rank', N=100)
    return recall, manual_efforts


if __name__ == '__main__':
    with open('', 'r') as f:
        cve_dict = json.load(f)
    with open('', 'r') as f:
        cve_multi_dict = json.load(f)

    print('1/7: start to prepare the dataset at', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    train_cve = cve_dict['train_cve']
    train_cve.extend(cve_multi_dict['train_cve'])
    print('len of train dataset', len(train_cve))
    train_data = NewDataset(train_cve, ns=1, ns_num=50)
    train_dataloader = DataLoader(dataset=train_data, shuffle=True, batch_size=32, num_workers=0)

    print('2/7: create a model at', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    model = NewModel().to(device)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    print('3/7: train the model at', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    result_dir = ''
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    train(model, train_dataloader, result_dir)

    print('4/7: start to prepare the test dataset at', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    test_cve = cve_dict['test_cve']
    test_cve.extend(cve_multi_dict['test_cve'])
    print('len of test dataset', len(test_cve), test_cve[0])
    test_data = NewDataset(test_cve)
    test_dataloader = DataLoader(dataset=test_data, batch_size=128, num_workers=0)

    print('5/7: test the model at', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    test_result_dir = ''
    if not os.path.exists(test_result_dir):
        os.makedirs(test_result_dir)

    model = NewModel()
    model.load_state_dict(torch.load(''))
    model = torch.nn.DataParallel(model).to(device) if MULTI_GPU else model.to(device)

    info = 'test'
    recall_100, manual_efforts_100 = test(model, test_dataloader, test_result_dir, info)
    print('*************** test ***************')
    print('recall_100: ', recall_100, '\nmanual_efforts_100: ', manual_efforts_100)

    print('6/7: start to prepare the validation dataset at', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
   
    test_cve = cve_dict['val_cve']
    test_cve.extend(cve_multi_dict['val_cve'])
    print('len of test dataset', len(test_cve), test_cve[0])
    test_data = NewDataset(test_cve)
    test_dataloader = DataLoader(dataset=test_data, batch_size=128, num_workers=0)

    print('7/7: validate the model at', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    info = 'validate'
    recall_100, manual_efforts_100 = test(model, test_dataloader, test_result_dir, info)
    print('*************** test ***************')
    print('recall_100: ', recall_100, '\nmanual_efforts_100: ', manual_efforts_100)
