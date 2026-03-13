import zipfile
import os

def extract_zip(zip_path, extract_to):

    # create folder if it doesn't exist
    if not os.path.exists(extract_to):
        os.makedirs(extract_to)

    # open the zip file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

    # return extracted file names
    return os.listdir(extract_to)