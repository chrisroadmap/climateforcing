"""Utility functions."""

import errno
import os
import urllib.request

from tqdm import tqdm


# This example is adapted from
# https://stackoverflow.com/questions/15644964/python-progress-bar-and-downloads
class _DownloadProgressBar(tqdm):
    def update_to(self, blocks=1, blocksize=1, totalsize=None):
        """Update download progress bar.

        Parameters
        ----------
            blocks : int, optional
                Number of blocks transferred so far [default: 1].
            blocksize : int, optional
                Size of each block (in tqdm units) [default: 1].
            totalsize : int, optional
                Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if totalsize is not None:
            self.total = totalsize
        self.update(blocks * blocksize - self.n)


def check_and_download(url, filepath="./", clobber=False):
    """Check prescence of a file and downloads if not present.

    Parameters
    ----------
        url : str
            url to download from
        filepath : str, optional, default="./"
            filename to download to. If a directory is specified, write to the
            directory using the filename from the URL.
        clobber : bool, default=False
            False if download should not overwrite an existing file, True if it
            should.
    """
    # if the target directory does not exist, create it
    target_dir = os.path.split(filepath)[0]
    mkdir_p(target_dir)

    # if a directory is specified as the filepath, download inside there.
    if os.path.isdir(filepath):
        filepath = f"{filepath}/{url.split('/')[-1]}"

    # do the download, if file doesn't exist or we are overwriting
    if clobber or not os.path.isfile(filepath):
        with _DownloadProgressBar(
            unit="B", unit_scale=True, miniters=1, desc=url.split("/")[-1]
        ) as progress:
            urllib.request.urlretrieve(
                url, filename=filepath, reporthook=progress.update_to
            )


# I got this from Stack Overflow, but can't find where now.
def mkdir_p(path):
    """Check to see if directory exists, and if not, create it.

    Parameters
    ----------
        path : str
            directory to create

    Raises
    ------
        OSError:
            if directory cannot be created
    """
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise
