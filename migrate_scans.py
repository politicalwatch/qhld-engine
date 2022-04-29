import json

from tipi_data import db
from tipi_data.models.scanned import Scanned

class P2030(db.DynamicDocument):
    meta = {'collection': 'scanned-p2030'}

def add_tags(origin, destination):
    kb = 'ods'
    destination.init_tagged_kb(kb)
    if 'tags' not in origin:
        return

    for tag in origin['tags']:
        destination.add_tag(kb, tag['topic'], tag['subtopic'], tag['tag'], tag['times'])

documents = P2030.objects()

for document in documents:
    scanned = Scanned(
        id=document['id'],
        title=document['title'],
        excerpt=document['excerpt'],
        created=document['created'],
        expiration=document['expiration'],
        verified=document['verified']
    )

    add_tags(document, scanned)
    scanned.save()
