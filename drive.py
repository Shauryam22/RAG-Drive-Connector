import io # input/op module , helps with downloading
import os 
from google.oauth2 import service_account  # to read credentials
from googleapiclient.discovery import build # to interact with G-API
from googleapiclient.http import MediaIoBaseDownload # to download in chunks
from dotenv import load_dotenv


load_dotenv() # to load the credentials/folder_id
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
SERVICE_ACCOUNT_FILE = os.getenv('SERVICE_FILE')
FOLDER_ID = os.getenv('FOLDER_ID')

if not FOLDER_ID or not SERVICE_ACCOUNT_FILE:
    raise ValueError('Folder id and service_acc is missing. Put it in .env')

def auth_drive():
    cred = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE,scopes = SCOPES)# fetching the credentials
    service = build('drive','v3',credentials=cred) # official google drive api client
    return service


def list_files(service,folder_id):
    query = f"'{folder_id}' in parents and trashed = false"
    results = service.files().list(
        q=query,spaces = 'drive',fields = 'nextPageToken, files(id, name)'
    ).execute() # search the query to find files inside the drive
    
    items = results.get('files',[])
    return items

def download_file(service, file_id, file_name, destination_folder="downloads"):
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
        
    request = service.files().get_media(fileId=file_id) 
    file_path = os.path.join(destination_folder, file_name)
    
    with open(file_path, 'wb') as fh:
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while done is False:
            status, done = downloader.next_chunk()
            print(f"Downloading {file_name}: {int(status.progress() * 100)}%")
            
    return file_path


if __name__ =='__main__':
    print('Authenticating....')
    drive_serv = auth_drive()
    files = list_files(drive_serv,FOLDER_ID)
    
    if not files:
        print('No files found')
    else:
        print('Files found')
        
        for file in files:
            print(f"{file['name']},{file['id']}")
            
