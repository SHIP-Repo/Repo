import gc
import itertools
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


os.environ['CUDA_VISIBLE_DEVICES'] = '5,6,7'
device = torch.device("cuda")
n_gpu = 3
num_epochs = 20
MULTI_GPU = 1

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


class NewPairDataset(Dataset):
    def __init__(self, feature_file):

        df_feature = pd.read_csv(feature_file)
        df_feature = df_feature.drop(
            ['msg_url1', 'msg_url2', 'diff_code1', 'diff_code2', 'commit_time1', 'commit_time2', 'author1', 'author2',
             'committer1', 'committer2'], axis=1)

        df_feature['msg_text1'] = df_feature['msg_text1'] + df_feature['deepseek_text1']
        df_feature['msg_text2'] = df_feature['msg_text2'] + df_feature['deepseek_text2']
        print('len of df:', len(df_feature))

        self.cve = df_feature['cve']
        self.commit1 = df_feature['commit1']
        self.commit2 = df_feature['commit2']
        self.msg_text1 = df_feature['msg_text1']
        self.msg_text2 = df_feature['msg_text2']
        self.label1 = df_feature['label1']
        self.label2 = df_feature['label2']
        self.label = df_feature['label']

       

        handcrafted_columns = ['cve_match', 'cve_num1', 'cve_num2', 'bug_match', 'bug_num1', 'bug_num2', 'issue_match',
                               'issue_num1', 'issue_num2', 'id_match', 'author_match', 'time_interval',
                               'same_func_used_num', 'same_func_used_ratio', 'opposite_ratio', 'opposite_num',
                               'same_ratio', 'same_num', 'same_function_num',
                               'same_function_ratio', 'same_file_num', 'same_file_ratio', 'same_msg_token_num',
                               'same_msg_token_ratio', 'commit_pair_tfidf', 'same_code_token_num', 'same_code_token_ratio',
                               'same_deepseek_text_token_num', 'same_deepseek_text_token_ratio',
                               'commit_pair_deepseek_tfidf', 'patch_score1', 'patch_score2'] 
        handcrafted_feature = df_feature[handcrafted_columns]
        self.handcrafted = handcrafted_feature.to_numpy()  

        self.text_tokenizer = RobertaTokenizer.from_pretrained('../pretrained_model/roberta-large')

        del df_feature
        gc.collect()

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        max_seq_length = 512
        msg_text1 = self.msg_text1[index] if isinstance(self.msg_text1[index], str) else ''
        msg_text2 = self.msg_text2[index] if isinstance(self.msg_text2[index], str) else ''
        input_ids_text, attention_mask_text = convert_examples_to_features(msg_text1, msg_text2, self.text_tokenizer,
                                                                           max_seq_length)

        sample = (input_ids_text, attention_mask_text, torch.tensor(self.handcrafted[index]),
                  torch.tensor(self.label[index]), self.cve[index], self.commit1[index], self.commit2[index],
                  self.label1[index], self.label2[index])

        return sample


class NewPairModel(nn.Module):
    def __init__(self):
        super(NewPairModel, self).__init__()

        self.hc_dim = 32    
        self.s_dim = 32
        config = RobertaConfig.from_pretrained('../pretrained_model/roberta-large')
        self.textEncoder = AutoModel.from_pretrained('../pretrained_model/roberta-large', config=config)

        self.fc1 = nn.Linear(self.textEncoder.config.hidden_size, self.s_dim)
        self.fc2 = nn.Linear(self.hc_dim, self.hc_dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.s_dim + self.hc_dim, (self.s_dim + self.hc_dim) // 2),
            nn.ReLU(),
            nn.Linear((self.s_dim + self.hc_dim) // 2, 2)
        )

        self.criterion = nn.CrossEntropyLoss()

        for param in self.textEncoder.parameters():
            param.requires_grad = True

    def forward(self, input_ids_text, attention_mask_text, handcrafted, label=None):
        text_output = self.textEncoder(input_ids=input_ids_text, attention_mask=attention_mask_text)[1] 

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
            handcrafted = batch[2].float().to(device)
            label = batch[3].long().to(device)

            loss, prob = model(input_ids_text, attention_mask_text, handcrafted, label)

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
            model_file = result_dir + ''
            model_to_save = model.module if hasattr(model, 'module') else model
            torch.save(model_to_save.state_dict(), model_file)

            full_model_file = result_dir + '/checkpoint_Phase2_full_model.bin'
            torch.save(model, full_model_file)

        torch.cuda.empty_cache()


def test(model, test_dataloader, result_dir, info):
    prob_list = []
    label_list = []
    cve_list = []
    commit1_list = []
    commit2_list = []
    label1_list = []
    label2_list = []

    model.eval()
    bar = tqdm(test_dataloader, total=len(test_dataloader))
    for step, batch in enumerate(bar):
        input_ids_text = batch[0].to(device)
        attention_mask_text = batch[1].to(device)
        handcrafted = batch[2].float().to(device)
        label = batch[3]
        cve = batch[4]
        commit1 = batch[5]
        commit2 = batch[6]
        label1 = batch[7]
        label2 = batch[8]

        with torch.no_grad():
            prob = model(input_ids_text, attention_mask_text, handcrafted)

            prob_list.append(prob.cpu().numpy())
            label_list.append(list(label))
            cve_list.append(list(cve))
            commit1_list.append(list(commit1))
            commit2_list.append(list(commit2))
            label1_list.append(list(label1))
            label2_list.append(list(label2))

    torch.cuda.empty_cache()
    cve_list = np.concatenate(cve_list, 0)
    prob_list = np.concatenate(prob_list, 0)
    prob_list = prob_list[:, 1]
    label_list = np.concatenate(label_list, 0)
    commit1_list = np.concatenate(commit1_list, 0)
    commit2_list = np.concatenate(commit2_list, 0)
    label1_list = np.concatenate(label1_list, 0)
    label2_list = np.concatenate(label2_list, 0)

    p_data = {
        'cve': cve_list,
        'commit1': commit1_list,
        'commit2': commit2_list,
        'predict': prob_list,
        'label': label_list,
        'label1': label1_list,
        'label2': label2_list
    }

    result_csv = pd.DataFrame(p_data)
    result_csv.to_csv('', index=False)

    return None


if __name__ == '__main__':

    print('5/7: test the model at', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    test_result_dir = ''
    if not os.path.exists(test_result_dir):
        os.makedirs(test_result_dir)

    model = NewPairModel()
    model.load_state_dict(torch.load(''))
    model = torch.nn.DataParallel(model).to(device) if MULTI_GPU else model.to(device)

    info = 'test'
    test(model, test_dataloader, test_result_dir, info)

