# import os
# import requests
# from bs4 import BeautifulSoup

# def get_file_ids_from_folder(folder_url):
#     """Retrieve file IDs and names from a public Google Drive folder URL."""
#     response = requests.get(folder_url)
#     if response.status_code != 200:
#         raise Exception("Failed to access the folder. Please check the URL.")

#     soup = BeautifulSoup(response.text, 'html.parser')
#     file_elements = soup.find_all('a', {'class': 'Q5txwe'})
#     files = []
#     for elem in file_elements:
#         file_id = elem.get('href').split('/')[-2]
#         file_name = elem.text
#         files.append((file_id, file_name))
#     return files

# def download_file(file_id, file_name, destination_folder):
#     """Download a single file by its file ID."""
#     download_url = f"https://drive.google.com/uc?id={file_id}&export=download"
#     response = requests.get(download_url, stream=True)
#     if response.status_code == 200:
#         os.makedirs(destination_folder, exist_ok=True)
#         file_path = os.path.join(destination_folder, file_name)
#         with open(file_path, 'wb') as f:
#             for chunk in response.iter_content(chunk_size=8192):
#                 f.write(chunk)
#         print(f"Downloaded: {file_name}")
#     else:
#         print(f"Failed to download: {file_name} (HTTP {response.status_code})")

# def download_folder(folder_url, destination_folder):
#     """Download all files from a public Google Drive folder."""
#     print("Fetching file list...")
#     files = get_file_ids_from_folder(folder_url)
#     print(f"Found {len(files)} files in the folder.")
#     for file_id, file_name in files:
#         download_file(file_id, file_name, destination_folder)

# if __name__ == '__main__':
#     # Replace with your public folder URL
#     folder_ID = '1-G8I5cJCD66fjpvejUzF9QPRJU_GNxj0'
#     folder_url = f'https://drive.google.com/drive/folders/{folder_ID}'
#     destination_folder = '~/team1/Ken/4DLangSplatSurgery/data'

#     download_folder(folder_url, destination_folder)

from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google_auth_oauthlib.flow import InstalledAppFlow
import io
import os

# Define the scopes
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

# Obtain your Google credentials
def get_credentials():
    flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
    creds = flow.run_local_server(port=0)
    return creds

# Build the downloader
creds = get_credentials()
drive_downloader = build('drive', 'v3', credentials=creds)

# Google Drive folder ID
folder_id = '1zTcX80c1yrbntY9c6-EK2W2UVESVEug8'

# query = f"Folder ID '{folder_id}'"  # you may get error for this line
query = f"'{folder_id}' in parents"  # this works  ref https://stackoverflow.com/q/73119251/248616

results = drive_downloader.files().list(q=query, pageSize=1000).execute()
items = results.get('files', [])

# Download the files
for item in items:
    request = drive_downloader.files().get_media(fileId=item['id'])
    f = io.FileIO(item['name'], 'wb')
    downloader = MediaIoBaseDownload(f, request)
    done = False
    while done is False:
        status, done = downloader.next_chunk()
        print(f"Download {int(status.progress() * 100)}.")

print(f"Downloaded {len(items)} files from the folder.")