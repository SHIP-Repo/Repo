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


os.environ['CUDA_VISIBLE_DEVICES'] = '5'
device = torch.device("cuda")
n_gpu = 1
num_epochs = 10
MULTI_GPU = 0

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
stemmer = nltk.stem.SnowballStemmer('english')
stop_words = set(stopwords.words('english'))
batch_size = 24

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


def collate_fn(batch):
    input_ids_batch = []
    attention_mask_batch = []
    hc_feature_batch = []
    label_batch = []
    cve_batch = []
    commit_batch = []

    for input_ids_text_list, attention_mask_text_list, hc_feature, label, cve, commit in batch:
        input_ids_batch.append(input_ids_text_list)
        attention_mask_batch.append(attention_mask_text_list)
        hc_feature_batch.append(hc_feature)
        label_batch.append(label)
        cve_batch.append(cve)
        commit_batch.append(commit)

    return (input_ids_batch, attention_mask_batch, hc_feature_batch, label_batch, cve_batch, commit_batch)


class NewDataset_RR(Dataset):
    def __init__(self, file_path):
        df = pd.read_csv(file_path)


        df['mess'] = df['msg_text'] + df['deepseek_text']
        handcrafted_columns = ['issue_cnt', 'bug_cnt', 'cve_cnt',
                               'cve_match', 'bug_match', 'issue_match', 'cwe_match',
                               'time_dis', 'vuln_type_1', 'vuln_type_2', 'vuln_type_3',
                               'filepath_same_cnt', 'filepath_same_ratio', 'filepath_unrelated_cnt',
                               'file_same_cnt', 'file_same_ratio', 'file_unrelated_cnt',
                               'func_same_cnt', 'func_same_ratio', 'func_unrelated_cnt',
                               'inter_token_cwe_cnt', 'inter_token_cwe_ratio', 'commit_vuln_tfidf',
                               'commit_vuln_ds_tfidf',
                               'mess_shared_num', 'mess_shared_ratio', 'mess_max', 'mess_sum', 'mess_mean', 'mess_var',
                               'ds_shared_num', 'ds_shared_ratio', 'ds_max', 'ds_sum', 'ds_mean', 'ds_var',
                               'patch_score']  
        df[handcrafted_columns] = df[handcrafted_columns].to_numpy()
        df['hc_feature'] = df[handcrafted_columns].apply(lambda row: row.tolist(), axis=1)

        grouped_data = []
        for cve, group in df.groupby('cve'):
            desc = group['desc'].tolist()[0]

            group_label1 = group[group['label'] == 1]
            commit_list = []
            mess_list = []
            hc_feature_list = []
            has_label1 = 0
            for _, row in group_label1.iterrows():
                commit_list.append(row['commit'])
                mess_list.append(row['mess'])
                hc_feature_list.append(row['hc_feature'])
                has_label1 = 1

            if has_label1 == 1:
                p_data = {
                    'cve': cve,
                    'commit': commit_list,
                    'desc': desc,
                    'mess': mess_list,
                    'hc_feature': hc_feature_list,
                    'label': 1
                }
                grouped_data.append(p_data)


            group_label0 = group[group['label'] == 0]
            for _, row in group_label0.iterrows():
                commit_list = [row['commit']]
                mess_list = [row['mess']]
                hc_feature_list = [row['hc_feature']]

                p_data = {
                    'cve': cve,
                    'commit': commit_list,
                    'desc': desc,
                    'mess': mess_list,
                    'hc_feature': hc_feature_list,
                    'label': 0
                }
                grouped_data.append(p_data)

        df = pd.DataFrame(grouped_data)

        self.cve = df['cve']
        self.commit = df['commit']
        self.desc = df['desc']
        self.mess = df['mess']
        self.handcrafted = df['hc_feature']
        self.label = df['label']

        self.text_tokenizer = RobertaTokenizer.from_pretrained('../pretrained_model/roberta-large')

        gc.collect()

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        max_seq_length = 512

        desc = self.desc[index] if isinstance(self.desc[index], str) else ''
        input_ids_text_list = []
        attention_mask_text_list = []
        for each in self.mess[index]:
            mess = each if isinstance(each, str) else ''
            input_ids_text, attention_mask_text = convert_examples_to_features(desc, mess, self.text_tokenizer,
                                                                               max_seq_length)
            input_ids_text_list.append(input_ids_text)
            attention_mask_text_list.append(attention_mask_text)


        sample = (input_ids_text_list, attention_mask_text_list, self.handcrafted[index],
                  self.label[index], self.cve[index], self.commit[index])

        return sample


class NewDataset_RR_test(Dataset):
    def __init__(self, file_path):
        df = pd.read_csv(file_path)

        df['mess'] = df['msg_text'] + df['deepseek_text']
        handcrafted_columns = ['issue_cnt', 'bug_cnt', 'cve_cnt',
                               'cve_match', 'bug_match', 'issue_match', 'cwe_match',
                               'time_dis', 'vuln_type_1', 'vuln_type_2', 'vuln_type_3',
                               'filepath_same_cnt', 'filepath_same_ratio', 'filepath_unrelated_cnt',
                               'file_same_cnt', 'file_same_ratio', 'file_unrelated_cnt',
                               'func_same_cnt', 'func_same_ratio', 'func_unrelated_cnt',
                               'inter_token_cwe_cnt', 'inter_token_cwe_ratio', 'commit_vuln_tfidf',
                               'commit_vuln_ds_tfidf',
                               'mess_shared_num', 'mess_shared_ratio', 'mess_max', 'mess_sum', 'mess_mean', 'mess_var',
                               'ds_shared_num', 'ds_shared_ratio', 'ds_max', 'ds_sum', 'ds_mean', 'ds_var',
                               'patch_score']  
      
        df[handcrafted_columns] = df[handcrafted_columns].to_numpy()
        df['hc_feature'] = df[handcrafted_columns].apply(lambda row: row.tolist(), axis=1)

        grouped_data = []
        for cve, group in df.groupby('cve'):
            desc = group['desc'].tolist()[0]

            for group_id, sub_group in group.groupby('group_id'):
                commit_list = []
                mess_list = []
                hc_feature_list = []
                for _, row in sub_group.iterrows():
                    commit_list.append(row['commit'])
                    mess_list.append(row['mess'])
                    hc_feature_list.append(row['hc_feature'])

                p_data = {
                    'cve': cve,
                    'commit': commit_list,
                    'desc': desc,
                    'mess': mess_list,
                    'hc_feature': hc_feature_list,
                    'group_id': group_id
                }
                grouped_data.append(p_data)

        df = pd.DataFrame(grouped_data)

        self.cve = df['cve']
        self.commit = df['commit']
        self.desc = df['desc']
        self.mess = df['mess']
        self.handcrafted = df['hc_feature']
        self.group_id = df['group_id']

        self.text_tokenizer = RobertaTokenizer.from_pretrained('../pretrained_model/roberta-large')

        gc.collect()

    def __len__(self):
        return len(self.group_id)

    def __getitem__(self, index):
        max_seq_length = 512

        desc = self.desc[index] if isinstance(self.desc[index], str) else ''
        input_ids_text_list = []
        attention_mask_text_list = []
        for each in self.mess[index]:
            mess = each if isinstance(each, str) else ''
            input_ids_text, attention_mask_text = convert_examples_to_features(desc, mess, self.text_tokenizer,
                                                                               max_seq_length)
            input_ids_text_list.append(input_ids_text)
            attention_mask_text_list.append(attention_mask_text)

        sample = (input_ids_text_list, attention_mask_text_list, self.handcrafted[index],
              self.group_id[index], self.cve[index], self.commit[index])

        return sample


class NewModel_RR(nn.Module):
    def __init__(self):
        super(NewModel_RR, self).__init__()

        self.batch_size = batch_size
        self.hc_dim = 37
        self.s_dim = 32
        config = RobertaConfig.from_pretrained('../pretrained_model/roberta-large')
        self.textEncoder = AutoModel.from_pretrained('../pretrained_model/roberta-large', config=config)

        self.fc1 = nn.Linear(self.textEncoder.config.hidden_size, self.s_dim)
        self.fc2 = nn.Linear(self.hc_dim, self.hc_dim)

        self.criterion = nn.CrossEntropyLoss()

        for param in self.textEncoder.parameters():
            param.requires_grad = True

    def forward(self, input_ids_text, attention_mask_text, handcrafted, group_info, label=None):
        text_output = self.textEncoder(input_ids=input_ids_text, attention_mask=attention_mask_text)[
            1] 
        text_output = self.fc1(text_output) 
        handcrafted = self.fc2(handcrafted)  
        combine_output = torch.cat([text_output, handcrafted], dim=-1)  
        hidden_vectors = self.mlp1(combine_output) 

        pooled_vectors = torch.zeros(len(group_info), hidden_vectors.size(-1)).to(device)

        for i, group in enumerate(group_info):
            group_vectors = hidden_vectors[group, :]  
            pooled_vectors[i, :] = torch.max(group_vectors, dim=0).values   

        logits = self.mlp2(pooled_vectors)

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
            input_ids_text_list = batch[0]
            attention_mask_text_list = batch[1]
            handcrafted_list = batch[2]
            label_list = batch[3]
            cve = batch[4]

            input_ids_text = []
            attention_mask_text = []
            handcrafted = []
            label = []
            group_info = []

            for i in range(len(input_ids_text_list)):
                group_info.append(list(range(len(input_ids_text), len(input_ids_text) + len(input_ids_text_list[i]))))
                input_ids_text.extend(input_ids_text_list[i])
                attention_mask_text.extend(attention_mask_text_list[i])
                label.append(label_list[i])
                handcrafted.extend(handcrafted_list[i])

                if (i + 1 < len(input_ids_text_list)) and (len(input_ids_text) + len(input_ids_text_list[i]) <= 16):
                    a = 1
                else:
                    input_ids_text = torch.stack(input_ids_text)
                    attention_mask_text = torch.stack(attention_mask_text)
                    handcrafted = torch.tensor(handcrafted)
                    handcrafted = handcrafted.float()
                    label = torch.tensor(label)

                    input_ids_text = input_ids_text.to(device)
                    attention_mask_text = attention_mask_text.to(device)
                    handcrafted = handcrafted.to(device)
                    label = label.long().to(device)

                    loss, prob = model(input_ids_text, attention_mask_text, handcrafted, group_info, label)

                    if n_gpu > 1:
                        loss = loss.mean() 
                    loss.backward()

                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    losses.append(loss.item())
                  
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()

                    input_ids_text = []
                    attention_mask_text = []
                    handcrafted = []
                    label = []
                    group_info = []

        print("epoch {} loss {}".format(idx, round(float(np.mean(losses)), 3)))

        if (idx + 1) % 2 == 0:
            print("---save model at epoch", idx)
            model_file = result_dir + '/checkpoint_Phase3_model-no-llm.bin'
            model_to_save = model.module if hasattr(model, 'module') else model
            torch.save(model_to_save.state_dict(), model_file)

            full_model_file = result_dir + '/checkpoint_Phase3_full_model-no-llm.bin'
            torch.save(model, full_model_file)

        torch.cuda.empty_cache()


def test(model, test_dataloader, result_dir, info):
    prob_list_test = []
    group_id_list_test  = []
    cve_list_test  = []
    commit_list_test  = []

    model.eval()
    bar = tqdm(test_dataloader, total=len(test_dataloader))
    for step, batch in enumerate(bar):
        input_ids_text_list = batch[0]
        attention_mask_text_list = batch[1]
        handcrafted_list = batch[2]
        group_id_list = batch[3]
        cve_list = batch[4]
        commit_list = batch[5]

        input_ids_text = []
        attention_mask_text = []
        handcrafted = []
        group_info = []
        group_id = []
        cve = []
        commit = []

        for i in range(len(input_ids_text_list)):
            group_info.append(list(range(len(input_ids_text), len(input_ids_text) + len(input_ids_text_list[i]))))
            input_ids_text.extend(input_ids_text_list[i])
            attention_mask_text.extend(attention_mask_text_list[i])
            handcrafted.extend(handcrafted_list[i])

            group_id.append(group_id_list[i])
            cve.append(cve_list[i])
            commit.append(commit_list[i])

            if (i + 1 < len(input_ids_text_list)) and (len(input_ids_text) + len(input_ids_text_list[i]) <= 64):
                a = 1
            else:
                input_ids_text = torch.stack(input_ids_text)
                attention_mask_text = torch.stack(attention_mask_text)
                handcrafted = torch.tensor(handcrafted)
                handcrafted = handcrafted.float()

                input_ids_text = input_ids_text.to(device)
                attention_mask_text = attention_mask_text.to(device)
                handcrafted = handcrafted.to(device)

                with torch.no_grad():
                    prob = model(input_ids_text, attention_mask_text, handcrafted, group_info)

                    prob_list_test.append(prob.cpu().numpy())
                    group_id_list_test.append(group_id)
                    cve_list_test.append(cve)
                    commit_list_test.append(commit)

                input_ids_text = []
                attention_mask_text = []
                handcrafted = []
                group_info = []
                group_id = []
                cve = []
                commit = []

    torch.cuda.empty_cache()
    cve_list_test = np.concatenate(cve_list_test, 0)
    prob_list_test = np.concatenate(prob_list_test, 0)
    prob_list_test = prob_list_test[:, 1]
    group_id_list_test = np.concatenate(group_id_list_test, 0)
    commit_list_test = np.concatenate(commit_list_test, 0)

    p_data = {
        'cve': cve_list_test,
        'commit_list': commit_list_test,
        'predict': prob_list_test,
        'group_id': group_id_list_test
    }
    result_csv = pd.DataFrame(p_data)
    result_csv.to_csv('', index=False)
    result_csv['rank'] = get_rank(result_csv, ['predict'], ascending=False)
    result_csv.to_csv('', index=False)
    result_csv['commit_list'] = result_csv['commit_list'].apply(eval)
    result_csv = result_csv.explode('commit_list', ignore_index=True)
    result_csv.rename(columns={'commit_list': 'commit'}, inplace=True)
    result_csv.to_csv('', index=False)

    df_test = pd.read_csv('')
    df_test = df_test.merge(result_csv, on=['cve', 'commit', 'group_id'], how='left')
    df_test.to_csv(result_dir + '', index=False)

    recall, manual_efforts = 0, 0
    return recall, manual_efforts


if __name__ == '__main__':
    print('1/7: start to prepare the dataset at', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    file_path = ''
    train_data = NewDataset_RR(file_path)
    train_dataloader = DataLoader(dataset=train_data, shuffle=False, batch_size=batch_size, num_workers=0, collate_fn=collate_fn)

    print('2/7: create a model at', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    model = NewModel_RR().to(device)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    print('3/7: train the model at', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    result_dir = ''
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    train(model, train_dataloader, result_dir)

    print('4/7: start to load model at', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    test_result_dir = ''
    if not os.path.exists(test_result_dir):
        os.makedirs(test_result_dir)

    model = NewModel_RR()
    model.load_state_dict(torch.load(''))
    model = torch.nn.DataParallel(model).to(device) if MULTI_GPU else model.to(device)

    print('5/8: start to prepare the test dataset at', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    file_path = ''
    test_data = NewDataset_RR_test(file_path)
    test_dataloader = DataLoader(dataset=test_data, batch_size=64, num_workers=0, collate_fn=collate_fn)

    print('6/8: test the model at', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    info = 'test'
    recall_100, manual_efforts_100 = test(model, test_dataloader, test_result_dir, info)
    print('*************** test ***************')
    print('recall_100: ', recall_100, '\nmanual_efforts_100: ', manual_efforts_100)

