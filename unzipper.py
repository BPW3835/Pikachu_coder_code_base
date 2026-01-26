import zipfile

with zipfile.ZipFile("OneDrive_2026-01-19.zip", "r") as zip_ref:
    for name in zip_ref.namelist():
        print("Extracting:", name)
    zip_ref.extractall("./extracted_files")
