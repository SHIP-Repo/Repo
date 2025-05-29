import pandas as pd
from git import Repo
from multiprocessing import Pool
import datetime
import os
from tqdm import tqdm
import re
import requests


input_csv_path = ''
output_csv_path = ''
progress_file = ''
error_log_file = ''
chunk_size = 10000
num_processes = 80

df_commit = pd.read_csv('')


def clean_string(s):
    if isinstance(s, str):
        return s.encode('utf-8', errors='ignore').decode('utf-8')
    return s

def log_error(message):
    with open(error_log_file, 'a') as f:
        f.write(f"{datetime.datetime.now()}: {message}\n")

def get_commit_info_from_github(repo, commit_hash, token):
    url = f"https://api.github.com/repos/{repo}/commits/{commit_hash}"
    headers = {
        "Accept": "application/vnd.github.v3+json"
    }
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        commit_data = response.json()
        message = commit_data['commit']['message']
        diff_url = commit_data['html_url'] + ".diff"
        diff_response = requests.get(diff_url, headers=headers)
        diff_code = diff_response.text if diff_response.status_code == 200 else "Failed to get diff"
        commit_time = datetime.datetime.strptime(commit_data['commit']['committer']['date'], "%Y-%m-%dT%H:%M:%SZ")
        author = commit_data['commit']['author']['name']
        committer = commit_data['commit']['committer']['name']

        return message, diff_code, commit_time, author, committer
    else:
        log_error(f"Failed to get commit info from github: {response.status_code} {response.text} {repo} {commit_hash}")
        return '', '', None, '', ''

def process_row(row):
    repo_name = row['repo']
    commit_hash = row['commit']
    repo_path = f'/mnt/gitrepo/{repo_name}'

    try:
        repo = Repo(repo_path)
        commit = repo.commit(commit_hash)
        message = clean_string(commit.message)
        commit_time = datetime.datetime.fromtimestamp(commit.committed_date)
        author = clean_string(commit.author.name if commit.author else 'Unknown')
        committer = clean_string(commit.committer.name if commit.committer else 'Unknown')
        try:
            if len(commit.parents) == 0:
                diff_code = repo.git.diff(commit_hash, ignore_blank_lines=True, ignore_space_at_eol=True)
            else:
                diff_code = repo.git.diff(f"{commit_hash}~1", commit_hash, ignore_blank_lines=True, ignore_space_at_eol=True)
        except Exception as e:
            print(f"Failed to get diff for commit {commit_hash} in repo {repo_path}: {e}")
            log_error(f"Failed to get diff for commit {commit_hash} in repo {repo_path}: {e}")
            diff_code = ''
    except Exception as e:
        message, diff_code, commit_time, author, committer = get_commit_info_from_github(repo_name.replace('_', '/'), commit_hash, '')

    sign_pattern = r'(?i)^\s*(Signed[-\s]off[-\s]by|Submitted[-\s]by|Reviewed[-\s]by|Reported-and-tested-by|Cc|Acked[-\s]by|Reported[-\s]by|Tested[-\s]by|Suggested[-\s]by|Co-developed-by):\s*([A-Za-z\s]+)\s*<'
    sign_name_matches = re.findall(sign_pattern, message, flags=re.MULTILINE)

    sign_name = [name.strip() for _, name in sign_name_matches]
    sign_name = list(set(sign_name))

    sign_line_pattern = r'(?i)^\s*(Signed[-\s]off[-\s]by|Submitted[-\s]by|Reviewed[-\s]by|Reported-and-tested-by|Cc|Acked[-\s]by|Reported[-\s]by|Tested[-\s]by|Suggested[-\s]by|Co-developed-by):.*'
    message = re.sub(sign_line_pattern, '', message, flags=re.MULTILINE)

    url_pattern = r'https?://[a-zA-Z0-9-._~:/?#[\]@!$&\'()*+,;%=]+'
    urls = re.findall(url_pattern, message)
    urls = list(set(urls))
    message = re.sub(url_pattern, '', message)

    message = message.replace('\r\n', ' ').replace('\n', ' ')

    if len(diff_code.split('\n')) > 1000:
        diff_code = '\n'.join(diff_code.split('\n')[:1000])
    diff_code = clean_string(diff_code)
    if diff_code == '':
        diff_code = ' '

    return {
        'cve': row['cve'],
        'repo': repo_name,
        'commit': commit_hash,
        'label': row['label'],
        'msg_text': message,
        'msg_url': urls,
        'msg_sign': sign_name,
        'diff_code': diff_code,
        'commit_time': commit_time,
        'author': author,
        'committer': committer
    }


df_iter = pd.read_csv(input_csv_path, encoding='utf-8', chunksize=chunk_size)

last_processed = 0
if os.path.exists(progress_file):
    with open(progress_file, 'r') as f:
        last_processed = int(f.read())

for i, df_chunk in enumerate(df_iter):
    if i < last_processed:
        continue  

    
    with Pool(num_processes) as pool:
        results = list(tqdm(pool.imap(process_row, df_chunk.to_dict('records')), total=len(df_chunk),
                            desc=f"Processing chunk {i + 1}"))

    new_data = [result for result in results if result is not None]

    chunk_df = pd.DataFrame(new_data)

    mode = 'a' if os.path.exists(output_csv_path) else 'w'
    header = False if mode == 'a' else True

    chunk_df.to_csv(output_csv_path, mode=mode, index=False, header=header, encoding='utf-8')

    with open(progress_file, 'w') as f:
        f.write(str(i + 1))

    print(f"Chunk {i + 1} processed and saved.")

print("All chunks processed.")
