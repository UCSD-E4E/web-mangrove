import os
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv

load_dotenv(verbose=True)

CONNECTION_STRING = os.getenv('CONNECTION_STRING')


class DirectoryClient:
    def __init__(self, connection_string, container_name):
        self.container_name = container_name
        # service_client = BlobServiceClient.from_connection_string(connection_string)
        # self.client = service_client.get_container_client(container_name)
        self.client = BlobServiceClient.from_connection_string(
            CONNECTION_STRING)

    # send big .zip file to blob just from filestream (no downloading)

    def create_blob_from_stream(self, blob_name, stream):
        print('in azure_blob from stream function')
        self.client.get_blob_client(
            self.container_name, blob_name).upload_blob(stream)
        return

    def upload_file(self, source, dest):
        '''
        Upload a single file to a path inside the container
        '''
        # print(f'Uploading {source} to {dest}')
        # do i need this with open thing??? not too sure bc i never use the data var
        with open(source, 'rb') as data:
            print(dest)
            self.client.get_blob_client(
                self.container_name, dest).upload_blob(data, overwrite=True)
        return

    def download(self, source, dest):
        '''
        Download a file or directory to a path on the local filesystem
        '''
        if not dest:
            raise Exception('A destination must be provided')

        blobs = self.ls_files(source, recursive=True)
        print('blobs to download:', blobs)
        if blobs:
            # if source is a directory, dest must also be a directory
            if not source == '' and not source.endswith('/'):
                source += '/'
            if not dest.endswith('/'):
                dest += '/'
            # append the directory name from source to the destination
            dest += os.path.basename(os.path.normpath(source)) + '/'

            blobs = [source + blob for blob in blobs]
            for blob in blobs:
                blob_dest = dest + os.path.relpath(blob, source)
                # print("\nDownloading blob to " + blob_dest)
                self.client.get_blob_client(self.container_name, blob).download_blob(
                ).download_to_stream(open(blob_dest, 'wb'))
        else:
            self.client.get_blob_client(self.container_name, blob).download_blob(
            ).download_to_stream(open(dest, 'wb'))

    def download_file(self, source, dest):
        '''
        # Download a single file to a path on the local filesystem
        '''
        # dest is a directory if ending with '/' or '.', otherwise it's a file
        if dest.endswith('.'):
            dest += '/'
        blob_dest = dest + \
            os.path.basename(source) if dest.endswith('/') else dest

        # print( f'Downloading {source} to {blob_dest}')
        os.makedirs(os.path.dirname(blob_dest), exist_ok=True)
        self.client.get_blob_client(self.container_name, source).download_blob(
        ).download_to_stream(open(blob_dest, 'wb'))
        '''data = bc.download_blob()
    file.write(data.readall())'''

    def ls_files(self, path, recursive=False):
        '''
        List files under a path, optionally recursively
        '''
        if not path == '' and not path.endswith('/'):
            path += '/'

        blob_iter = self.client.get_container_client(
            self.container_name).list_blobs(path)

        files = []
        for blob in blob_iter:
            relative_path = os.path.relpath(blob.name, path)
            if recursive or not '/' in relative_path:
                files.append(relative_path)
        return files

    def ls_dirs(self, path, recursive=False):
        '''
        List directories under a path, optionally recursively
        '''
        if not path == '' and not path.endswith('/'):
            path += '/'

        blob_iter = self.client.get_container_client(
            self.container_name).list_blobs(path)

        dirs = []
        for blob in blob_iter:
            relative_dir = os.path.dirname(os.path.relpath(blob.name, path))
            if relative_dir and (recursive or not '/' in relative_dir) and not relative_dir in dirs:
                dirs.append(relative_dir)

        return dirs

    def rm(self, path, recursive=False):
        '''
        Remove a single file, or remove a path recursively
        '''
        if recursive:
            self.rmdir(path)
        else:
            print(f'Deleting {path}')
            self.client.get_blob_client(
                self.container_name, path).delete_blob()

    def rmdir(self, path):
        '''
        Remove a directory and its contents recursively
        '''
        blobs = self.ls_files(path, recursive=True)
        if blobs == []:
            return

        if not path == '' and not path.endswith('/'):
            path += '/'
        blobs = [path + blob for blob in blobs]
        print(f'Deleting blobs in ' + str(self.container_name))
        for blob in blobs:
            self.client.get_blob_client(
                self.container_name, blob).delete_blob()
