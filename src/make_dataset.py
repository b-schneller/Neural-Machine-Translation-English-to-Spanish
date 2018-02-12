from urllib.request import urlopen
import tarfile
import os


def make_dataset(args):
    corpus_url = 'http://www.statmt.org/europarl/v7/es-en.tgz'

    if not os.path.isdir(args.raw_data_directory):
        os.makedirs(args.raw_data_directory)

    corpus_destination = args.raw_data_directory
    print('Downloading and unzipping datasets...')
    download_unzip_files(corpus_url, corpus_destination)


def download_unzip_files(tar_url, destination):
    tar_resp = urlopen(tar_url)
    temp_tar = open('/tmp/tempfile.tgz', 'wb')
    temp_tar.write(tar_resp.read())
    temp_tar.close()
    tar_file = tarfile.open('/tmp/tempfile.tgz', 'r')
    tar_file.extractall(path=destination)
