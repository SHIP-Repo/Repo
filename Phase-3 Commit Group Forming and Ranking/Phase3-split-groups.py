import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import difflib


threshold = 0.1


dataset = ''
data_dir_path = ''      

df = pd.read_csv(data_dir_path + "commit-pair_score_{}.csv".format(dataset))
df['is_related'] = df['predict'].apply(lambda score: 1 if score >= threshold else 0)


all_connected_components = {}
commit_in_group_rows = []
for cve, group in df.groupby('cve'):
    related_group = group[group['is_related'] == 1]

    G = nx.Graph()                      
    commit1_list = group['commit1'].to_list()
    commit2_list = group['commit2'].to_list()
    all_commits = list(set(commit1_list + commit2_list))
    G.add_nodes_from(all_commits)       
    for _, row in group.iterrows():     
        if row['is_related'] == 1:
            G.add_edge(row['commit1'], row['commit2'])
    connected_components = list(nx.connected_components(G))  
    all_connected_components[cve] = connected_components

    component_mapping = {}  
    for i, component in enumerate(connected_components, start=1):
        for commit in component:
            component_mapping[commit] = i

  
    commit1_unique = group[['cve', 'commit1', 'label1']].drop_duplicates()
    commit2_unique = group[['cve', 'commit2', 'label2']].drop_duplicates()

    commit1_unique.columns = ['cve', 'commit', 'label']
    commit2_unique.columns = ['cve', 'commit', 'label']

    commit_in_group = pd.concat([commit1_unique, commit2_unique], ignore_index=True)
    commit_in_group = commit_in_group.drop_duplicates()
    commit_in_group['group_id'] = commit_in_group['commit'].apply(lambda x: component_mapping.get(x, 0))

    commit_in_group = commit_in_group.sort_values(by='group_id', ascending=True)  
    commit_in_group = pd.concat(
        [commit_in_group[commit_in_group['group_id'] != 0], commit_in_group[commit_in_group['group_id'] == 0]])
    commit_in_group = commit_in_group.reset_index(drop=True)

    commit_in_group_rows.append(commit_in_group)


df_connected_components = pd.DataFrame(
    [(cve, str(connected_components)) for cve, connected_components in all_connected_components.items()],
    columns=['cve', 'connected_components']
)
df_connected_components.to_csv(
    data_dir_path + "commit-pair_score_{}_connected_components.csv".format(dataset), index=False)

df_group_id = pd.concat(commit_in_group_rows, ignore_index=True)
df_group_id.to_csv("../data/Phase3_input_data/commit-pair_score_{}-commit_group-id-0.1.csv".format(dataset), index=False)
file_path = '../data/Phase3_input_data/Phase3-test-feature.csv'
df10 = pd.read_csv(file_path)
df10=df10.drop('group_id',axis=1)
df10=df10.merge(df_group_id,on=['cve','commit','label'])
df10.to_csv("../data/Phase3_input_data/Phase3-test-feature-0.1.csv",index=False)


