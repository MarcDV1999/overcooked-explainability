import pandas as pd
import sys
import os

sys.path.append("./Code")


def join_results(partial_file, complete_file):
    partial_df = pd.read_csv(partial_file)
    complete_df = pd.read_csv(complete_file)

    new_columns = {name: 'Partial ' + name for name in partial_df.columns[2:]}
    partial_df.rename(columns=new_columns, inplace=True)

    new_columns = {name: 'Complete ' + name for name in complete_df.columns[2:]}
    complete_df.rename(columns=new_columns, inplace=True)

    partial = partial_df[partial_df.columns[1:]]
    complete = complete_df[complete_df.columns[1:]]

    df = pd.merge(partial, complete, left_on=partial.columns[0], right_on=complete.columns[0])
    df = df.astype(float)
    df[df.columns[0]] = df[df.columns[0]].astype(int)
    df[df.columns[3]] = df[df.columns[3]].astype(int)
    df[df.columns[9]] = df[df.columns[9]].astype(int)

    return df


def select_file(new_list):
    print("Files and directories in '", results_folder, "':")
    for identifier, file in enumerate(new_list):
        if not file.find('Results'):
            print(f'\t+ {file}')
        else:
            print(f'\t{identifier}: {file}')
    return int(input('Select Partial file: '))


def select_folder(new_list):

    print("Directories in '", results_folder, "':")
    for id, file in enumerate(new_list):
        if not file.find('Results'):
            print(f'\t+ {file}')
        else:
            print(f'\t{id}: {file}')
    return int(input('Select Folder: '))


if __name__ == '__main__':

    # Select which layout
    layout = 'random3'
    results_folder = f'MDP_Results/{layout}/'

    dir_list = list(
        filter(
            lambda file_name: True if os.path.isdir(results_folder + file_name) else False, os.listdir(results_folder)
        )
    )
    dir_list.sort()

    # Experiment Folder
    folder_id = select_folder(dir_list)
    results_folder = results_folder + dir_list[folder_id] + '/'

    dir_list = list(filter(lambda file_name: True if file_name.endswith('.csv') else False, os.listdir(results_folder)))
    dir_list.sort()

    # Partial
    new_list = [file_name for file_name in dir_list if file_name.find('P') >= 1]
    partial_id = select_file(new_list)
    partial_file = results_folder + new_list[partial_id]

    # Complete
    new_list = [file_name for file_name in dir_list if file_name.find('C') >= 1]
    complete_id = select_file(new_list)
    complete_file = results_folder + new_list[complete_id]

    df = join_results(partial_file, complete_file)
    discretizer = partial_file.split('_')[-1]
    df.to_csv(results_folder + 'Results_' + discretizer)
