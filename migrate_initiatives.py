import json

from tipi_data import db
from tipi_data.models.initiative import Initiative
from tipi_data.models.alert import Alert, Search
from tipi_data.utils import generate_id


class Tipi(db.DynamicDocument):
    meta = {'collection': 'initiatives-tipi'}

class P2030(db.DynamicDocument):
    meta = {'collection': 'initiatives-p2030'}

class TipiAlerts(db.DynamicDocument):
    meta = {'collection': 'alerts-tipi'}

class P2030Alerts(db.DynamicDocument):
    meta = {'collection': 'alerts-p2030'}

sources = [
    {
        'initiatives': P2030,
        'alerts': P2030Alerts,
        'knowledgebase': 'parlamento2030'
    },
    {
        'initiatives': Tipi,
        'alerts': TipiAlerts,
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
    for source in sources:
        collection = source['initiatives']
        kb = source['knowledgebase']
        print('Merging initiatives from knowledgebase ' + kb)

        initiatives = collection.objects()

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
        initiative = Initiative.all.get(id=id)
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
    destination.init_tagged_kb(kb)
    if 'tags' not in origin:
        return

    for tag in origin['tags']:
        destination.add_tag(kb, tag['topic'], tag['subtopic'], tag['tag'], tag['times'])

def __get_value(f, o):
    try:
        return o[f]
    except KeyError:
        return None

def merge_and_delete_answers():
    answers = Initiative.objects(initiative_type_alt='Respuesta')
    for answer in answers:
        print("Processing {}".format(answer['reference']))
        question = Initiative.objects(reference=answer['reference'], initiative_type_alt__ne='Respuesta').first()
        question['extra']['answer'] = dict()
        question['extra']['answer']['content'] = answer['content']
        question['extra']['answer']['tagged'] = [a.serialize() for a in answer['tagged']]
        question.save()
        answer.delete()

search_fields = [
        'hash',
        'search',
        'dbsearch',
        'created',
        'validated',
        'validation_email_sent',
        'validation_email_sent_date',
        ]

def merge_alerts():
    for source in sources:
        collection = source['alerts']
        kb = source['knowledgebase']
        print('Merging alerts from knowledgebase ' + kb)

        alerts = collection.objects()
        count = alerts.count()
        counter = 0
        for new_alert in alerts:
            counter += 1
            print(str(counter) + ' out of ' + str(count))

            alert = get_alert(new_alert['email'])

            for alert_search in new_alert['searches']:
                search = copy_search(alert_search, kb)

                if 'searches' not in alert:
                    alert['searches'] = []

                alert['searches'].append(search)
            alert.save()

def copy_search(source, kb):
    search = json.loads(source['search'])
    search['knowledgebase'] = kb
    return Search(
            hash=source['hash'],
            search=json.dumps(search),
            dbsearch=source['dbsearch'],
            created=source['created'],
            validated=source['validated'],
            validation_email_sent=source['validation_email_sent'],
            validation_email_sent_date=source['validation_email_sent_date']
            )

loaded_alerts = {}
def get_alert(email):
    email = email.replace('gmail-com', 'gmail.com')

    if email in loaded_alerts:
        return loaded_alerts[email]

    try:
        alert = Alert.objects().get(email=email)
    except Exception:
        alert = Alert(
            id=generate_id(email),
            email=email
        )
    loaded_alerts[email] = alert
    return alert

def execute_merge():
    merge_initiatives()
    #merge_and_delete_answers()
    merge_alerts()

execute_merge()
