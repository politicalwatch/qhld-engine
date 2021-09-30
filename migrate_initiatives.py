from tipi_data import db
from tipi_data.models.initiative import Initiative


class Tipi(db.DynamicDocument):
    meta = {'collection': 'initiatives-tipi'}

class P2030(db.DynamicDocument):
    meta = {'collection': 'initiatives-p2030'}

collections = [
    {
        'collection': P2030,
        'knowledgebase': 'parlamento2030'
    },
    {
        'collection': Tipi,
        'knowledgebase': 'tipiciudadano'
    },
]

fields = [
    'id',
    'reference',
    'title',
    'initiative_type',
    'initiative_type_alt',
    'author_deputies',
    'author_others',
    'author_parliamentarygroups',
    'place',
    'created',
    'updated',
    'history',
    'status',
    'url',
    'content',
    'extra',
]

def merge_initiatives():
    for collection in collections:
        db_collection = collection['collection']
        kb = collection['knowledgebase']

        initiatives = db_collection.objects()

        for new_initiative in initiatives:
            initiative = get_initiative(new_initiative['id'])

            for field in fields:
                set_field(field, new_initiative, initiative)

            add_tags(kb, new_initiative, initiative)

            initiative.save()

loaded_initiatives = {}
def get_initiative(id):
    if id in loaded_initiatives:
        return loaded_initiatives[id]

    try:
        initiative= Initiative.all.get(id=id)
    except Exception:
        initiative = Initiative(id=id)

    loaded_initiatives[id] = initiative
    return initiative

def set_field(field, origin, destination):
    exist_on_origin = field in origin
    empty_on_destination = field not in destination or destination[field] == None or not destination[field]
    if exist_on_origin and empty_on_destination:
        destination[field] = origin[field]

def add_tags(kb, origin, destination):
    if 'tags' not in origin:
        return

    for tag in origin['tags']:
        destination.add_tag(kb, True, tag['topic'], tag['subtopic'], tag['tag'], tag['times'])

merge_initiatives()
