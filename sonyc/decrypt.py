import pycurl
import base64
import h5py
import tarfile
import os
import io
import resampy
import soundfile as psf

from urllib.parse import urlencode
from Crypto.Cipher import AES


def decrypt(url,
            cacert,
            cert,
            key,
            encrypted_key_filebuf=None,
            encrypted_data_filebuf=None,
            encrypted_key_filepath=None,
            encrypted_data_filepath=None,
            output_file=None, verbose=False):
    """
    Decrypt a file or filebuffer.

    Parameters
    ----------
    url : str
        Decryption server URL
    cacert : str
        Path to CA cert
    cert : str
        Path to your cert
    key : str
        Path to key file
    encrypted_key_filebuf : BytesIO
        Buffer of encrypted data key
    encrypted_data_filebuf : BytesIO
        Buffer of encrypted data (audio)
    encrypted_key_filepath : str
        Path to encrypted data key
    encrypted_data_filepath : str
        Path to encrypted data (audio)
    output_file : str
        Path to save file to
    verbose : bool
        If True, print info

    Returns
    -------
    decrypted_data : BytesIO
    """
    # make sure that either encrypted_key buffer or file is defined
    if encrypted_key_filebuf is None and encrypted_key_filepath is None:
        raise Exception('Either `encrypted_key` or `encrypted_key_file` must be defined.')

    # make sure that enithe encrypted_data buffer or file is defined
    if encrypted_data_filebuf is None and encrypted_data_filepath is None:
        raise Exception('Either `encrypted_key` or `encrypted_key_file` must be defined.')

    buf = io.BytesIO()

    c = pycurl.Curl()
    c.setopt(c.POST, True)
    c.setopt(c.URL, url)
    c.setopt(c.WRITEDATA, buf)

    c.setopt(pycurl.CAINFO, cacert)

    c.setopt(pycurl.SSLCERTTYPE, "PEM")
    c.setopt(pycurl.SSLCERT, cert)

    c.setopt(pycurl.SSLKEYTYPE, "PEM")
    c.setopt(pycurl.SSLKEY, key)

    c.setopt(pycurl.SSL_VERIFYPEER, 1)
    c.setopt(pycurl.SSL_VERIFYHOST, 2)

    if encrypted_key_filepath is not None:
        with open(encrypted_key_filepath, mode='r') as f:
            encrypted_key_filebuf = f.read()

    if encrypted_data_filepath is not None:
        with open(encrypted_data_filepath, mode='rb') as f:
            encrypted_data_filebuf = f.read()

    post_data = {
        "out_format": "raw",
        "enc": encrypted_key_filebuf
    }

    postfields = urlencode(post_data)

    c.setopt(c.POSTFIELDS, postfields)

    c.perform()

    if verbose:
        print('Status: %d' % c.getinfo(c.RESPONSE_CODE))
        print('Request Time: %f' % c.getinfo(c.TOTAL_TIME))

    decrypt_key = buf.getvalue()

    cipher = AES.new(decrypt_key)

    decrypted_data = cipher.decrypt(
        base64.b64decode(encrypted_data_filebuf)).rstrip(b'{')

    if output_file is not None:
        with open(output_file, 'wb') as f:
            f.write(decrypted_data)

    return io.BytesIO(decrypted_data)


def read_encrypted_tar_audio_file(enc_tar_filepath, enc_tar_filebuf=None, sample_rate=None, **kwargs):
    """
    Given the tarfile (or buffer of a tarfile) of a recording, untar and decrypt

    Parameters
    ----------
    enc_tar_filepath : str
        This is required even if enc_tar_filebuf is given, since we need to extract the name from enc_tar_filepath
    enc_tar_filebuf : File Obj
    sample_rate : int
    kwargs : dict
        `decrypt` arguments

    Returns
    -------
    y : np.array
        Audio data
    sr : int
        Sample rate of `y`
    identifier : str
        Recording identifier
    """
    adir = enc_tar_filepath.replace('.tar.gz', '')

    identifier = os.path.basename(adir)

    if enc_tar_filebuf is None:
        tar = tarfile.open(enc_tar_filepath, mode='r:gz')
    else:
        tar = tarfile.open(fileobj=enc_tar_filebuf, mode='r:gz')

    enc = dict()
    for member in tar.getmembers():
        enc[os.path.splitext(member.name)[1]] = tar.extractfile(member)

    # decrypt
    buf = decrypt(encrypted_key_filebuf=enc['.key'].read(), encrypted_data_filebuf=enc['.enc'].read(), **kwargs)

    y, sr = psf.read(buf)

    if sample_rate is not None and sr != sample_rate:
        y = resampy.resample(y, sr, sample_rate, filter='kaiser_fast')
    else:
        sample_rate = sr

    return y, sample_rate, identifier


def read_encrypted_tar_audio_file_from_day_tar(day_tar_filepath, enc_tar_filename, sample_rate, **kwargs):
    """
    Extract and decrypt a single file from a day's tar file, all using buffers

    Parameters
    ----------
    day_tar_filepath : str
    enc_tar_filename : str
    sample_rate : int
    kwargs : dict
        `decrypt` arguments

    Returns
    -------
    y : np.array
        Audio data
    sr : int
        Sample rate
    identifier : str
        Recording identifier
    """
    # open tar file
    tar = tarfile.open(day_tar_filepath)

    # get tar member info
    member = tar.getmember(enc_tar_filename)

    # extract recording file from day file
    enc_tar_filebuf = tar.extractfile(member)

    return read_encrypted_tar_audio_file(enc_tar_filename,
                                         enc_tar_filebuf=enc_tar_filebuf,
                                         sample_rate=sample_rate,
                                         **kwargs)


def read_encrypted_tar_audio_file_from_day_hdf5(day_hdf5_filepath,
                                                enc_tar_filename,
                                                enc_tar_index,
                                                sample_rate,
                                                **kwargs):
    """
    Extract and decrypt a single file from a day's tar file, all using buffers

    Parameters
    ----------
    day_hdf5_filepath : str
    enc_tar_filename : str
    sample_rate : int
    enc_tar_index : int

    kwargs : dict
        `decrypt` arguments

    Returns
    -------
    y : np.array
        Audio data
    sr : int
        Sample rate
    identifier : str
        Recording identifier
    """
    # open hdf5 file
    with h5py.File(day_hdf5_filepath, 'r') as h5:
        d = h5['recordings']
        # find row
        if enc_tar_index is not None:
            data = d[enc_tar_index]['data']
        else:
            data = d[d['filename'] == enc_tar_filename]['data']
        # assert(d[enc_tar_index]['filename'].decode('utf-8')==enc_tar_filename)

    # extract recording file from day file
    enc_tar_filebuf = io.BytesIO(data)

    return read_encrypted_tar_audio_file(enc_tar_filename.decode('utf-8'),
                                         enc_tar_filebuf=enc_tar_filebuf,
                                         sample_rate=sample_rate,
                                         **kwargs)
