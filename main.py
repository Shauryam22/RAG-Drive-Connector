import os
from dotenv import load_dotenv

# Importing from your existing files!
from drive import auth_drive, list_files, download_file
from parser import process_doc

load_dotenv(override=True)
FOLDER_ID = os.getenv("FOLDER_ID")


def run_sync_pipeline():
    print("Starting automated sync pipeline...")

    drive_serv = auth_drive()
    files = list_files(drive_serv, FOLDER_ID)

    if not files:
        print("No files found in the folder.")
        return []

    all_processed_chunks = []

    for file in files:
        current_file_id = file["id"]
        current_file_name = file["name"]

        print(f"\n--- Processing: {current_file_name} ---")

        local_path = download_file(drive_serv, current_file_id, current_file_name)
        chunks = process_doc(local_path, current_file_id, current_file_name)

        if chunks:
            all_processed_chunks.extend(chunks)

    print(
        f"\nPipeline Complete! Successfully processed into {len(all_processed_chunks)} total chunks."
    )
    #print(all_processed_chunks)
    return all_processed_chunks

# print(run_sync_pipeline)
