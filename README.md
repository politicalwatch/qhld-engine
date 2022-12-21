# tipi-engine
Motor de tareas y procesos internos de Tipi

## Requirements
Install setup and it will install all dependencies and libraries
```
./setup.sh
```

## Configuration

All variables are in .env file


Add to Crontab /etc/crontab
=======
```
0 2	*/3 * * root	bash /path/to/cron.sh
```

Reset Denylist
=======
Access to redis-cli and flush:
```
flushall
```
If you want only flush a specific db:
```
select [number db]
flushdb
```

Available Commands
=======

## Extractor
- `python quickex.py extractor initiatives`: Extracts new initiatives.
- `python quickex.py extractor references`: Calculates references that are not present in the database and prints them.
- `python quickex.py extractor votes`: Extracts new votes.
- `python quickex.py extractor interventions`: Extracts new video interventions.
- `python quickex.py extractor all-initiatives`: Extracts all the initiatives.
- `python quickex.py extractor all-references`: Calculates all the references.
- `python quickex.py extractor all-votes`: Extracts all the votes.
- `python quickex.py extractor all-interventions`: Extracts all the video interventions
- `python quickex.py extractor single-initiative [initiative reference]`: Extracts (or updates) a single initiative, untagging it if it already existed
- `python quickex.py extractor single-intervention [initiative reference]`: Extracts the video interventions of a single reference.
- `python quickex.py extractor single-vote [initiative reference]`: Extracts the votes of a single reference.
- `python quickex.py extractor type-initiative [initiative code]`: Extracts all the new initiatives of the specified initiative type.
- `python quickex.py extractor type-references`: Prints all the new initiatives of the specified initiative type.
- `python quickex.py extractor type-interventions`: Extracts all the video interventions of the new initiatives of the specified initiative type.
- `python quickex.py extractor type-votes`: Extracts all the votes of the new initiatives of the specified initiative type.
- `python quickex.py extractor type-all-initiative [initiative code]`: Extracts all the initiatives of the specified initiative type.
- `python quickex.py extractor type-all-references`: Prints all the initiatives of the specified initiative type.
- `python quickex.py extractor type-all-interventions`: Extracts all the video interventions of the initiatives of the specified initiative type.
- `python quickex.py extractor type-all-votes`: Extracts all the votes of the initiatives of the specified initiative type.

- `python quickex.py extractor load-groups "[PARLIAMENTARY_GROUPS_FILE.JSON]"`: Initiatizes all the parliamentary groups (it must be executed once at the beginning of a legislature).
- `python quickex.py extractor members`: Extracts all the members and updates the existing ones in the DB.
- `python quickex.py extractor calculate-composition-groups`: Calculates composition (based on members) for all the parliamentary groups.

## Tagger
- `python quickex.py tagger all` (default): Tags all the initiatives with all the tags and topics.
- `python quickex.py tagger all-long`: Tags all the long-content initiatives with all the tags and topics.
- `python quickex.py tagger kb "[KNOWLEDGE_BASE]"`: Tags all the initiatives with all the tags and topics of the specified knowledge base.
- `python quickex.py tagger new-topic "[TOPIC]"`: Tags all the initiatives with the all tags of the specified topic. Must be already present in the topic dictionary.
- `python quickex.py tagger new-tag "[TOPIC]" "[TAG]"`: Tags all the initiatives with the tag specified. Must be already present in the topics dictionary.
- `python quickex.py tagger modify-regex  "[TOPIC]" "[TAG]"`: Finds the tag and removes it from all initiatives and tags all the initiatives using the updated regex. The regex must be updated in the topic dictionary.
- `python quickex.py tagger rename-tag  "[TOPIC]" "[OLD TAG]" "[NEW TAG]"`: Finds all the occurrences of the specified tag in all the initiatives and replace it with the new name.
- `python quickex.py tagger reference "[REFERENCE]"`: Tags the specified initiative

## Untagger
- `python quickex.py untagger all` (default): Marks all initiatives as not tagged.
- `python quickex.py untagger kb [KNOWLEDGE_BASE]`: Removes tags from the specified knowledge base from all initiatives.
- `python quickex.py untagger topic "[TOPIC]"`: Removes all tags from the specified topic from all initiatives.
- `python quickex.py untagger tag  "[TOPIC]" "[TAG]"`: Removes the specified topic's tag from all the initiatives.
- `python quickex.py untagger reference "[REFERENCE]"`: Removes all the tags from the specified initiative.

## Alerts
- `python quickex.py send-alerts: Send the alerts.
- `python quickex.py generate-alert [REFERENCE]`: Inserts the specified initiative inside the initiative_alerts collection.
