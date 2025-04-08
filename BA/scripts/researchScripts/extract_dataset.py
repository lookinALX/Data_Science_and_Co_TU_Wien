import os
import tarfile

def safe_extract_filter(member, numeric_owner=False):
    member.uname = member.gname = ""
    return member

def extract_tar(file_path: str, path_to_extract_directory: str):
    file_name = os.path.basename(file_path)

    os.makedirs(path_to_extract_directory, exist_ok=True)

    try:
        with tarfile.open(file_path, "r:gz") as tar:
            print(f"Extraction of {file_name} has been started.")
            tar.extractall(path=path_to_extract_directory, filter=safe_extract_filter)
        print(f"Extraction of {file_name} completed successfully.")
    except Exception as e:
        print(f"Extraction of {file_name} failed: {e}")


def read_path_file(file_path: str):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.readline().strip()


if __name__ == "__main__":
    current_directory = os.getcwd()
    parent_directory = os.path.dirname(current_directory)
    data_folder = os.path.join(parent_directory, 'data')

    path_to_extract_directory = read_path_file(os.path.join(data_folder, 'extraction_path.txt'))

    for root, dirs, files in os.walk(data_folder):
        for file in files:
            if file.endswith('.tar.gz'):
                file_path = os.path.join(root, file)
                extract_tar(file_path, path_to_extract_directory)