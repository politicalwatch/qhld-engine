import sys

from extractors.extractor import ExtractorTask
from tagger.tag_initiatives import TagInitiatives
from untagger.untag_initiatives import UntagInitiatives
from alerts.send_alerts import SendAlerts
from stats.process_stats import GenerateStats
from tipi_data.repositories.initiatives import Initiatives
from tipi_data.models.alert import create_alert


def print_help():
    print('Usage: quickex.py TASK')
    print('Apply task: alerts, tagger, untagger, stats or extractor')
    print('Example: python quickex.py stats')

def run_command(commands, arguments):
    if len(arguments) > 2:
        if arguments[2] in commands:
            args_amount = len(arguments)
            if args_amount == 4:
                return commands[arguments[2]](arguments[3])
            if args_amount == 5:
                return commands[arguments[2]](arguments[3], arguments[4])
            else:
                return commands[arguments[2]]()
        else:
            print('quickex: invalid TASK')
            return
    commands['default']()

def send_alerts(args):
    SendAlerts()

def generate_alert(arguments):
    reference = arguments[2]
    initiative = Initiatives.by_reference(reference)
    create_alert(initiative[0])
    print('Alerts created')


def modify_regex(tag):
    tagger = TagInitiatives()
    untagger = UntagInitiatives()

    untagger.remove_tag(tag)
    tagger.new_tag(tag)


def tag(args):
    command = TagInitiatives()
    subcommands = {
        'all': command.run,
        'all-long': command.tag_long,
        'kb': command.tag_kb,
        'new-tag': command.new_tag,
        'new-topic': command.new_topic,
        'modify-regex': modify_regex,
        'rename-tag': command.rename,
        'reference': command.by_reference,
        'default': command.run
    }
    run_command(subcommands, args)

def stats(args):
    GenerateStats().generate()

def untag(args):
    command = UntagInitiatives()
    subcommands = {
        'all': command.untag_all,
        'kb': command.by_kb,
        'topic': command.by_topic,
        'tag': command.by_tag,
        'reference': command.by_reference,
        'default': command.untag_all
    }
    run_command(subcommands, args)

def extract(args):
    task = ExtractorTask()
    subcommands = {
        'initiatives': task.initiatives,
        'references': task.references,
        'votes': task.votes,
        'interventions': task.interventions,
        'all-initiatives': task.all_initiatives,
        'all-references': task.all_references,
        'all-votes': task.all_votes,
        'all-interventions': task.all_interventions,
        'single-initiative': task.single_initiatives,
        'single-intervention': task.single_interventions,
        'single-vote': task.single_votes,
        'type-initiative': task.type_initiatives,
        'type-references': task.type_references,
        'type-interventions': task.type_interventions,
        'type-votes': task.type_votes,
        'type-all-initiative': task.type_all_initiatives,
        'type-all-references': task.type_all_references,
        'type-all-interventions': task.type_all_interventions,
        'type-all-votes': task.type_all_votes,
        'members': task.members,
        'default': task.run
    }
    run_command(subcommands, args)

def long_questions(args):
    query = {
                'content.100000': {'$exists': True}
            }
    initiatives = Initiatives.by_query(query)
    lengths = {}

    selected_initiatives = []
    tagged_initiative = False
    for initiative in initiatives:
        content = initiative['content']
        content_str = "".join(content)
        content_len = len(content_str)
        if content_len > 531800 and content_len < 532100:
            if not tagged_initiative and all(tagged.has_topics for tagged in initiative.tagged) and len(initiative.tagged) == 2:
                tagged_initiative = initiative
            selected_initiatives.append(initiative)

    total = len(selected_initiatives)
    counter = 1
    for initiative in selected_initiatives:
        print(f"Copying tags to initiative {initiative['reference']} {initiative['id']}. {counter} out of {total}")
        initiative['tagged'] = tagged_initiative['tagged']
        initiative.save()
        counter += 1



commands = {
    'alerts': send_alerts,
    'generate-alert': generate_alert,
    'tagger': tag,
    'untagger': untag,
    'stats': stats,
    'extractor': extract,
    'long-questions': long_questions,
}

args = sys.argv
if len(args) > 1:
    if args[1] in commands:
        commands[args[1]](args)
    else:
        print('quickex: invalid TASK')
        print_help()
else:
    print('quickex: bad number of params')
    print_help()
