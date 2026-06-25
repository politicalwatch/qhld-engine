import os
import hashlib

FILES = [
    'extractor.tmp',
    'tagger.tmp',
    'alerts.tmp',
    ]

def clean_files():
    for file in FILES:
        os.remove(file)


def generateId(*args):
    try:
        return hashlib.sha1(
                u''.join(args).encode('utf-8')
                ).hexdigest()
    except:
        return 'ID_ERROR'
