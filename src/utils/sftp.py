import os, stat

import paramiko
from dotenv import load_dotenv


class SFTPHelper:
    def __init__(self):
        load_dotenv()

        # Get data from .env file
        self.transport = paramiko.Transport((os.getenv("NAS_HOST"), int(os.getenv("NAS_PORT"))))
        self.transport.connect(username=os.getenv("NAS_USER"), password=os.getenv("NAS_PASS"))
        self.sftp = paramiko.SFTPClient.from_transport(self.transport)

    def exists(self, path):
        try:
            self.sftp.stat(path)
            return True
        except IOError:
            return False

    def is_dir(self, path):
        try:
            return stat.S_ISDIR(self.sftp.stat(path).st_mode)
        except IOError:
            return False

    def get_size(self, path):
        return self.sftp.stat(path).st_size

    def makedirs(self, remote_directory):
        """Recreates 'os.makedirs' for SFTP."""
        dirs = remote_directory.split('/')
        current_dir = ""
        if remote_directory.startswith('/'):
            current_dir = "/"
        for part in dirs:
            if not part: continue
            current_dir = os.path.join(current_dir, part)
            if not self.exists(current_dir):
                self.sftp.mkdir(current_dir)

    def listdir(self, path):
        return [file_attr.filename for file_attr in self.sftp.listdir_attr(path)]

    def download(self, remote_path, local_path):
        self.sftp.get(remote_path, local_path)

    def close(self):
        self.sftp.close()
        self.transport.close()



