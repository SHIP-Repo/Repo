import re
import ast
import pandas as pd
from huggingface_hub.utils import tqdm

dataset = ''
df1 = pd.read_csv('')

df2=pd.read_csv('')
df1['commit']=df1['commit'].apply(eval)
df2['commit']=df2['commit'].apply(eval)
df1=df1[df1['commit'].apply(lambda x:len(x)>1)]
df2=df2[df2['commit'].apply(lambda x:len(x)>1)]
for index,row in tqdm(df1.iterrows(), total=len(df1)):
    text = row['answer']
    match = re.search(r'({.*?})', text, re.DOTALL)
    if match:
        json_str = match.group(1)

        match = re.search(r'"code_change_summary":\s*"([^"]*)"', text)
        if match:
            summarization = match.group(1)
            df1.at[index, 'summarization'] = summarization
        match = re.search(r'"new_msg":\s*"([^"]*)"', text)
        if match:
            msg = match.group(1)
            df1.at[index, 'msg_text'] = msg

        match = re.search(r'"addressed_vulnerability_types":\s*(\[[^\]]*\])', text)
        if match:
            potential_vulnerability_types = match.group(1)

            if '+' in potential_vulnerability_types:
                potential_vulnerability_types = potential_vulnerability_types.replace('+', '')
                print(index)
            potential_vulnerability_types = ast.literal_eval(potential_vulnerability_types)
            df1.at[index, 'potential_addressed_vulnerability_types'] = str(potential_vulnerability_types)
        else:
            df1.at[index, 'potential_addressed_vulnerability_types'] = '[]'
        match = re.search(r'"is_patch":\s*"([^"]*)"', text)
        if match:
            is_patch = match.group(1)
            df1.at[index, 'is_patch'] = is_patch

df2['msg_text'] = df1['msg_text']
df1.to_csv('', index=False)
df2.to_csv('',index=False)
df_deepseek = df1[(df1['is_patch'] == 'YES/NO/UNKNOWN') | (df1['is_patch'].isna())]

