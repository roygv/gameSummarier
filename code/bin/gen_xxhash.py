#!/usr/bin/env python3

import os
import sys
import glob
import argparse
from urllib import request
import xxhash
from xxhash import xxh64

# https://pypi.python.org/pypi/xxhash/

def calculate_hash(file_path, chunks=0, sparse=False):
    hasher = xxh64()
    #print("=== opening: ", file_path)
    with open(file_path, 'rb') as f: # binary, read-only
        #print("=== opened: ", f)
        count=0
        eof=False
        while True:
            #print(" ....reading ", count)
            buf = f.read(4096)
            count = count + 1
            if not buf:
                eof=True
                #print(" ....done")
                break
            if chunks and count > chunks:
                #print("limit hash to first 4096 *", chunks)
                break
            hasher.update(buf)
        if(sparse and not eof):
            f.seek(-4096, os.SEEK_END)
            buf = f.read()
            if buf:
                hasher.update(buf)
                #print("including file tail in hash")
    return hasher.hexdigest()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate fast (partial) file hash.')
    parser.add_argument('files', metavar='files', type=str, nargs='+',
                        help='files to generate hash')
    parser.add_argument('-c', '--chunks', metavar='N', type=int, nargs=1,
                        help='limit hash to just first N chunks of file (chunks of 4096 bytes)')
    parser.add_argument('-s', '--sparse', action='store_true',
                        help='quick hash using just first & last of file')
    
    args = parser.parse_args()
    chunks = args.chunks[0]  if(args.chunks) else 0
    if(args.sparse and not chunks):
        chunks=256  # read 4096 * 256 = 1MB, plus last 4KB of file

    #print("chunks:", chunks)
    #print("args.chunks:", args.chunks)
    #print("args.sparse:", args.sparse)
    #print("args.files:", args.files)
    #print("============(begin) {}".format(os.getcwd()))


    for arg in args.files:
        #print("\n====== arg: {}".format(arg))
        # merge glob w/ original string, in case glob returns none (if the filename has special chars)
        for f in sorted(set(list(glob.glob(arg)) + [arg])):
            #print("=== file: ", f)
            xhash = calculate_hash(f, chunks=chunks, sparse=args.sparse)
            print("{}\t{}".format(xhash, f))

    #print("============(done).")
