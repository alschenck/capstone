import zipfile
import os

down_dir = r'C:\dev\download'
unzip_dir = r'C:\dev\download\unzip'


def unzip():
    for file in os.listdir(down_dir):
        if len(file.split(".")) > 1 and file.split(".")[1].lower() == "zip":
            zfile = zipfile.ZipFile(down_dir + "\\" + file)
            zfile.extractall(unzip_dir)

if __name__ == '__main__':
    unzip()
