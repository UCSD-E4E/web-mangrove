import os
from azure.storage.blob import BlockBlobService

account_name = 'mangroveclassifier'   # Azure account name
account_key = 's0T0RoyfFVb/Efc+e/s1odYn2YuqmspSxwRW/c5IrQcH5gi/FpHgVYpAinDudDQuXdMFgrha38b0niW6pHzIFw=='      # Azure Storage account access key  

class DirectoryClient:
  def __init__(self, connection_string, container_name):
    self.container_name = container_name
    # service_client = BlobServiceClient.from_connection_string(connection_string)
    # self.client = service_client.get_container_client(container_name)
    self.client = BlockBlobService(account_name=account_name, account_key=account_key)
    
    
  # send big .zip file to blob just from filestream (no downloading)
  def create_blob_from_stream(self, blob_name, stream):
    print('in azure_blob from stream function')
    self.client.create_blob_from_stream(self.container_name, blob_name, stream)
    return 
  
  def upload_file(self, source, dest):
    '''
    Upload a single file to a path inside the container
    '''
    # print(f'Uploading {source} to {dest}')
    # do i need this with open thing??? not too sure bc i never use the data var
    with open(source, 'rb') as data:
      self.client.create_blob_from_path(self.container_name, dest, source)
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
        self.client.get_blob_to_path(self.container_name, blob, blob_dest)
    else:
      self.client.get_blob_to_path(self.container_name, source, dest)

  def download_file(self, source, dest):
    '''
    # Download a single file to a path on the local filesystem
    '''
    # dest is a directory if ending with '/' or '.', otherwise it's a file
    if dest.endswith('.'):
      dest += '/'
    blob_dest = dest + os.path.basename(source) if dest.endswith('/') else dest

    # print( f'Downloading {source} to {blob_dest}')
    os.makedirs(os.path.dirname(blob_dest), exist_ok=True)
    self.client.get_blob_to_path(self.container_name, source, blob_dest)
    '''data = bc.download_blob()
    file.write(data.readall())'''

 

  def ls_files(self, path, recursive=False):
    '''
    List files under a path, optionally recursively
    '''
    if not path == '' and not path.endswith('/'):
      path += '/'
    
    blob_iter = self.client.list_blobs(self.container_name, prefix=path)
  
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

    blob_iter = self.client.list_blobs(self.container_name, prefix=path)
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
      self.client.delete_blob(path)

  def rmdir(self, path):
    '''
    Remove a directory and its contents recursively
    '''
    blobs = self.ls_files(path, recursive=True)
    if blobs==[]:
      return

    if not path == '' and not path.endswith('/'):
      path += '/'
    blobs = [path + blob for blob in blobs]
    print(f'Deleting blobs in ' + str(self.container_name))
    for blob in blobs: 
      self.client.delete_blob(blob_name=blob, container_name=self.container_name)
    

    