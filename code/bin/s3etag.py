#!/usr/bin/env python3
import sys
import hashlib

def calculate_s3_etag(file_path, chunk_size=8 * 1024 * 1024):
    '''
    Generate the s3 e-tag, which is used when an object is uploaded using 
    the multi-part API instead of the single file upload API (automatically
    imposed on larger files and/or if using the s3api vs the s3 client.)
    '''
    # see also: https://forums.aws.amazon.com/thread.jspa?messageID=203510
    # http://docs.aws.amazon.com/cli/latest/reference/s3api/head-object.html

    md5s = []

    with open(file_path, 'rb') as fp:
        while True:
            data = fp.read(chunk_size)
            if not data:
                break
            md5s.append(hashlib.md5(data))

    if len(md5s) == 1:
        return '"{}"'.format(md5s[0].hexdigest())

    digests = b''.join(m.digest() for m in md5s)
    digests_md5 = hashlib.md5(digests)
    return '"{}-{}"'.format(digests_md5.hexdigest(), len(md5s))


if __name__ == '__main__':
    for file_path in sys.argv[1:]:
        print(file_path, calculate_s3_etag(file_path))

