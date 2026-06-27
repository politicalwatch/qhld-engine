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

Commands are exposed through the `qhld` console script (installed with the
package, e.g. `uv sync`).

## Extractor
- `qhld extractor initiatives`: Extracts new initiatives.
- `qhld extractor references`: Calculates references that are not present in the database and prints them.
- `qhld extractor votes`: Extracts new votes.
- `qhld extractor interventions`: Extracts new video interventions.
- `qhld extractor all-initiatives`: Extracts all the initiatives.
- `qhld extractor all-references`: Calculates all the references.
- `qhld extractor all-votes`: Extracts all the votes.
- `qhld extractor all-interventions`: Extracts all the video interventions
- `qhld extractor single-initiative [initiative reference]`: Extracts (or updates) a single initiative, untagging it if it already existed
- `qhld extractor single-intervention [initiative reference]`: Extracts the video interventions of a single reference.
- `qhld extractor single-vote [initiative reference]`: Extracts the votes of a single reference.
- `qhld extractor type-initiative [initiative code]`: Extracts all the new initiatives of the specified initiative type.
- `qhld extractor type-references`: Prints all the new initiatives of the specified initiative type.
- `qhld extractor type-interventions`: Extracts all the video interventions of the new initiatives of the specified initiative type.
- `qhld extractor type-votes`: Extracts all the votes of the new initiatives of the specified initiative type.
- `qhld extractor type-all-initiative [initiative code]`: Extracts all the initiatives of the specified initiative type.
- `qhld extractor type-all-references`: Prints all the initiatives of the specified initiative type.
- `qhld extractor type-all-interventions`: Extracts all the video interventions of the initiatives of the specified initiative type.
- `qhld extractor type-all-votes`: Extracts all the votes of the initiatives of the specified initiative type.

- `qhld extractor load-groups "[PARLIAMENTARY_GROUPS_FILE.JSON]"`: Initializes all the parliamentary groups (it must be executed once at the beginning of a legislature).
- `qhld extractor members`: Extracts all the members and updates the existing ones in the DB.
- `qhld extractor calculate-composition-groups`: Calculates composition (based on members) for all the parliamentary groups.

## Tagger
- `qhld tagger all` (default): Tags all the initiatives with all the tags and topics.
- `qhld tagger all-long`: Tags all the long-content initiatives with all the tags and topics.
- `qhld tagger amendments`: Tags all the amendments with all the tags and topics.
- `qhld tagger kb "[KNOWLEDGE_BASE]"`: Tags all the initiatives with all the tags and topics of the specified knowledge base.
- `qhld tagger new-topic "[TOPIC]"`: Tags all the initiatives with the all tags of the specified topic. Must be already present in the topic dictionary.
- `qhld tagger new-tag "[TOPIC]" "[TAG]"`: Tags all the initiatives with the tag specified. Must be already present in the topics dictionary.
- `qhld tagger modify-regex  "[TOPIC]" "[TAG]"`: Finds the tag and removes it from all initiatives and tags all the initiatives using the updated regex. The regex must be updated in the topic dictionary.
- `qhld tagger rename-tag  "[TOPIC]" "[OLD TAG]" "[NEW TAG]"`: Finds all the occurrences of the specified tag in all the initiatives and replace it with the new name.
- `qhld tagger reference "[REFERENCE]"`: Tags the specified initiative

## Untagger
- `qhld untagger all` (default): Marks all initiatives as not tagged.
- `qhld untagger kb [KNOWLEDGE_BASE]`: Removes tags from the specified knowledge base from all initiatives.
- `qhld untagger topic "[TOPIC]"`: Removes all tags from the specified topic from all initiatives.
- `qhld untagger tag  "[TOPIC]" "[TAG]"`: Removes the specified topic's tag from all the initiatives.
- `qhld untagger reference "[REFERENCE]"`: Removes all the tags from the specified initiative.

## Alerts
- `qhld send-alerts: Send the alerts.
- `qhld debug generate-alert [REFERENCE]`: Inserts the specified initiative inside the initiative_alerts collection.

## Stats
- `qhld stats: Calculates stats.

## Footprint
- `qhld footprint: Calculates footprint.

## Topic alignment
- `qhld topic-alignment [INITIATIVE ID]: Calculates topic alignment for all (or one) initiative.
