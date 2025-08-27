import os
import gdown

def organize_folder_into_subfolders(path_to_original_folder, max_number_of_files_per_subfolder=50):
    '''Moves all files in a folder into newly created subfolders comprising of the max_number_of_files_per_subfolder or fewer'''
    files_in_folder = os.listdir(path_to_original_folder)
    if not path_to_original_folder.endswith('/'):
        path_to_original_folder += '/'
    temp_path_to_original_folder = path_to_original_folder + 'temp_folder'
    os.makedirs(temp_path_to_original_folder)
    subfolders_dict = {'temp_subfolder_0': []}
    os.makedirs(temp_path_to_original_folder + '/' + 'temp_subfolder_0')
    for _file_name in files_in_folder:
        if len(subfolders_dict['temp_subfolder_' + str(len(subfolders_dict) - 1)]) == max_number_of_files_per_subfolder:
            subfolders_dict['temp_subfolder_' + str(len(subfolders_dict))] = []
            os.makedirs(temp_path_to_original_folder + '/' + 'temp_subfolder_' + str(len(subfolders_dict) - 1))
        subfolders_dict['temp_subfolder_' + str(len(subfolders_dict) - 1)].append(_file_name)
    for _file_subfolder_path, _file_names in subfolders_dict.items():
        for _file_name in _file_names:
            os.rename(path_to_original_folder + _file_name, temp_path_to_original_folder + '/' + _file_subfolder_path + '/' + _file_name)
    return subfolders_dict

def undo_organize_folder_into_subfolders(path_to_original_folder, path_to_new_folder, subfolders_dict):
    '''Moves the files organized as subfolders back to the original & new folders and deletes subfolders'''
    if not path_to_original_folder.endswith('/'):
        path_to_original_folder += '/'
    if not path_to_new_folder.endswith('/'):
        path_to_new_folder += '/'
    temp_path_to_original_folder = path_to_original_folder + 'temp_folder'
    temp_path_to_new_folder = path_to_new_folder + 'temp_folder'
    for _file_subfolder_path, _file_names in subfolders_dict.items():
        for _file_name in _file_names:
            os.rename(temp_path_to_original_folder + '/' + _file_subfolder_path + '/' + _file_name, path_to_original_folder + _file_name)
            os.rename(temp_path_to_new_folder + '/' + _file_subfolder_path + '/' + _file_name, path_to_new_folder + _file_name)
        os.rmdir(temp_path_to_original_folder + '/' + _file_subfolder_path)
        os.rmdir(temp_path_to_new_folder + '/' + _file_subfolder_path)
    os.rmdir(temp_path_to_original_folder)
    os.rmdir(temp_path_to_new_folder)

def download_folder(folder_ID:str):
    url = f'https://drive.google.com/drive/folders/{folder_ID}=drive_link'
    gdown.download_folder(url, quiet=True, use_cookies=False, remaining_ok=True)

folder_ID = '1zTcX80c1yrbntY9c6-EK2W2UVESVEug8'
#path = '~/team1/Ken/4DLangSplatSurgery/data/hypernerf'
path = '~/team1/data/EndoNeRF'
subfolders_dict = organize_folder_into_subfolders(path)
download_folder(folder_ID)
undo_organize_folder_into_subfolders(path, subfolders_dict)
