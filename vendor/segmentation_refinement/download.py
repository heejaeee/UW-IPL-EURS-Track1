import os
import hashlib


def download_model(destination):
    """Automatic downloads are disabled in this offline reproduction package."""
    raise FileNotFoundError(
        f'CascadePSP requires a valid local weight file: {destination}. '
        'Fetch and extract the archive with repro_check.py --download-weights '
        'before starting offline inference. No automatic download is attempted.'
    )


def check_model(destination):
    correct_md5 = '7478d4a9c42ab52beb6d7e9683402fe0'

    digest = hashlib.md5()
    with open(destination, 'rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            digest.update(chunk)
    md5_returned = digest.hexdigest()

    return correct_md5 == md5_returned


def download_and_or_check_model_file(destination):

    if not os.path.isfile(destination):
        download_model(destination)
    if not check_model(destination):
        raise ValueError(
            f'CascadePSP local weight checksum mismatch: {destination}. '
            'Restore the verified weight archive before offline inference; '
            'the invalid file has not been modified.'
        )

