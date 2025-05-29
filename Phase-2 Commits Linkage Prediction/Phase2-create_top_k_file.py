import pandas as pd

df_val = pd.read_csv('')
df_each = pd.read_csv('')
df = df_val.merge(df_each[['cve', 'commit', 'label', 'msg_text', 'msg_url', 'diff_code', 'commit_time']], how='inner', on=['cve', 'commit', 'label'])
df = df.reset_index(drop=True)
    
df_each = pd.read_csv('')
merge_df = df_val.merge(df_each[['cve', 'commit', 'label', 'msg_text', 'msg_url', 'diff_code', 'commit_time']], how='inner', on=['cve', 'commit', 'label'])
df = pd.concat([df, merge_df], ignore_index=True)

df_val = pd.read_csv('')
df_each = pd.read_csv('')
df = df_val.merge(df_each[['cve', 'commit', 'label', 'msg_text', 'msg_url', 'diff_code', 'commit_time']], how='inner', on=['cve', 'commit', 'label'])
df = df.reset_index(drop=True)

for i in range(4):       
    df_each = pd.read_csv('')
    merge_df = df_val.merge(df_each[['cve', 'commit', 'label', 'msg_text', 'msg_url', 'diff_code', 'commit_time']], how='inner', on=['cve', 'commit', 'label'])
    df = pd.concat([df, merge_df], ignore_index=True)
data_df_val = df
data_df_val.to_csv('', index=False)

df_cve = pd.read_csv('')
df = pd.read_csv('')
df = df.merge(df_cve[['cve', 'desc', 'cwe']], how='left', on=['cve'])
df.to_csv('', index=False)

dataset = ''
df = pd.read_csv("")
df_deepseek = pd.read_csv("")

df_deepseek = df_deepseek.dropna(subset=['is_patch'])
df_deepseek = df_deepseek[df_deepseek['is_patch'] != 'YES/NO/UNKNOWN']

mapping = {'YES': 1, 'UNKNOWN': 0.5, 'NO': 0}
df_deepseek['patch_score'] = df_deepseek['is_patch'].map(mapping)

df_deepseek['potential_addressed_vulnerability_types'].fillna('[]', inplace=True)
df_deepseek['potential_addressed_vulnerability_types'] = df_deepseek['potential_addressed_vulnerability_types'].apply(eval)
df_deepseek['potential_addressed_vulnerability_types_text'] = df_deepseek['potential_addressed_vulnerability_types'].apply(
    lambda x: f"It may address {', '.join(x)}." if x else "")
df_deepseek['deepseek_text'] = df_deepseek['summarization'] + df_deepseek['potential_addressed_vulnerability_types_text']

df_deepseek = df_deepseek[['cve', 'commit', 'deepseek_text', 'patch_score']]
df = df.merge(df_deepseek, left_on=['cve', 'commit1'], right_on=['cve', 'commit'], how='inner')
df.rename(columns={'deepseek_text': 'deepseek_text1'}, inplace=True)
df.rename(columns={'patch_score': 'patch_score1'}, inplace=True)
df = df.merge(df_deepseek, left_on=['cve', 'commit2'], right_on=['cve', 'commit'], how='inner')
df.rename(columns={'deepseek_text': 'deepseek_text2'}, inplace=True)
df.rename(columns={'patch_score': 'patch_score2'}, inplace=True)
df.to_csv('')
