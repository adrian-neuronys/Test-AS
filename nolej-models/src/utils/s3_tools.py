import logging
import boto3
import os, sys
import dotenv

logger = logging.getLogger('neuronys.' + __name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.abspath(os.path.join(BASE_DIR, '../../config.env'))
#dotenv.load_dotenv(CONFIG_PATH)

# = boto3.Session(aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
#                            aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY'])

def list_bucket_content(bucket_name):
    """
    List the content of an s3 bucket
    Args:
        bucket_name:

    Returns:

    """
    s3 = boto3.resource('s3')
    bucket = s3.Bucket(bucket_name)
    keys_list = []

    for obj in bucket.objects.all():
        key = obj.key
        keys_list.append(key)

    return keys_list

def split_s3_url(s3_url):
    """
    Splits an Amazon S3 url such as s3://bucketname/filekey into a tuple (bucketname, filekey)
    Args:
        s3_url:

    Returns:

    """
    if s3_url.startswith('s3://'):
        s3_url = s3_url[5:]  # cut off 's3://'

    if '/' not in s3_url:
        raise ValueError("Invalid S3 URL: {}. Format must be [s3://]bucket/filekey".format(s3_url))

    return s3_url.split('/', 1)


def download_file(s3, bucket, filekey, destfile, log=True):
    '''download file from s3 bucket to disk
    :param s3: an S3 object, e.g. from s3 = boto3.resource('s3')

    implements a possibility to auto-fix the access rights on a file (ACL) when environment variable
    'ACL_AUTOFIX_ENABLED' is set to True (also see bucket_tools.set_acl())'''
    if log: print('S3 download to: ' + destfile)

    path_elements = destfile.split("/")
    current_position = "/"

    for folder in path_elements[:-1]:
        if folder!="":
            current_position = os.path.join(current_position, folder)
            if not os.path.exists(current_position):
                os.mkdir(current_position)

    s3.Bucket(bucket).download_file(filekey, destfile)

    return destfile

def download_s3_folder(key, local_dir="/tmp/", exclude_checkpoint_folders=True):
    """
    Download the contents of a folder directory
    Args:
        bucket_name: the name of the s3 bucket
        s3_folder: the folder path in the s3 bucket
        local_dir: a relative or absolute directory path in the local file system
    """
    bucket_name, s3_folder = split_s3_url(key)
    print(s3_folder)
    #s3 = session.resource('s3')
    s3 = boto3.resource("s3")

    bucket = s3.Bucket(bucket_name)
    for obj in bucket.objects.filter(Prefix=s3_folder):
        if exclude_checkpoint_folders:
            if "checkpoint" in obj.key:
                continue
        file_target_in_folder = obj.key.replace(s3_folder,"")
        if file_target_in_folder.startswith("/"):
            file_target_in_folder = file_target_in_folder[1:]
        target = os.path.join(local_dir, "_".join(s3_folder.split('/')),file_target_in_folder)
        if not os.path.exists(os.path.dirname(target)):
            os.makedirs(os.path.dirname(target))
        if obj.key[-1] == '/':
            continue
        if not os.path.exists(target):
            logger.info("Downloading {} to {}".format(obj.key, target))
            bucket.download_file(obj.key, target)

    return os.path.join(local_dir, "_".join(s3_folder.split('/')))



def s3_ls(url, return_all_keys=False):
    url = url.replace("s3://","")
    if url.endswith("/"):
        url = url[:-1]
    splitted_url = url.split("/")
    bucket = splitted_url[0]
    prefix = ("/".join(splitted_url[1:]))

    all_keys = list(get_matching_s3_keys(bucket, prefix))
    all_keys = [item[:-1] if item.endswith('/') else item for item in all_keys]
    if return_all_keys:
        return all_keys
    else:
        # Returns keys that are only in the specified folder
        depth_url = len(prefix.split('/'))

        return list({os.path.join("s3://",bucket,'/'.join(item.split('/')[:depth_url+1])) for item in all_keys})


def get_matching_s3_objects(bucket, prefix="", suffix=""):
    """
    Generate objects in an S3 bucket.

    :param bucket: Name of the S3 bucket.
    :param prefix: Only fetch objects whose key starts with
        this prefix (optional).
    :param suffix: Only fetch objects whose keys end with
        this suffix (optional).
    """
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")

    kwargs = {'Bucket': bucket}

    # We can pass the prefix directly to the S3 API.  If the user has passed
    # a tuple or list of prefixes, we go through them one by one.
    if isinstance(prefix, str):
        prefixes = (prefix, )
    else:
        prefixes = prefix

    for key_prefix in prefixes:
        kwargs["Prefix"] = key_prefix

        for page in paginator.paginate(**kwargs):
            try:
                contents = page["Contents"]
            except KeyError:
                break

            for obj in contents:
                key = obj["Key"]
                if key.endswith(suffix):
                    yield obj


def get_matching_s3_keys(bucket, prefix="", suffix=""):
    """
    Generate the keys in an S3 bucket.

    :param bucket: Name of the S3 bucket.
    :param prefix: Only fetch keys that start with this prefix (optional).
    :param suffix: Only fetch keys that end with this suffix (optional).
    """
    for obj in get_matching_s3_objects(bucket, prefix, suffix):
        yield obj["Key"]