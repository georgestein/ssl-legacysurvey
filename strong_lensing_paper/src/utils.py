import os
import wget

def list_files(startpath):
    """Print directory structure and files within
    Taken from https://stackoverflow.com/questions/9727673/list-directory-tree-structure-in-python
    """
    print(f'{startpath} structure and files:')

    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print('{}{}/'.format(indent, os.path.basename(root)))
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            print('{}{}'.format(subindent, f))


def get_h5py_file(filein, url_path, download=True):
    """Check for h5py version of filein after removing file extension
    
    Download from input URL if it does not exist
    """
    path = os.path.splitext(filein)[0]
    path_h5 = path+'.h5'
    
    local_h5 = os.path.exists(path_h5)
    if local_h5:
        print(f"h5py data file already exists at:\n {path_h5}")
    else:        
        file_url = os.path.join(url_path, os.path.basename(path_h5))
        print(f"h5py data file does not exist. Downloading from:\n {file_url}")
        
        if download:
            wget.download(file_url, path_h5)
    
