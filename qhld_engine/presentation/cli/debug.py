"""`qhld debug` — ad-hoc maintenance commands, run by hand (not by the DAGs)."""

import typer

app = typer.Typer(help="Ad-hoc, run-by-hand maintenance commands.")


@app.command("generate-alert")
def generate_alert(reference: str):
    """Create a 'Nueva iniciativa' alert for the initiative with the given reference."""
    from tipi_data.repositories.alerts import InitiativeAlerts
    from tipi_data.repositories.initiatives import Initiatives

    initiative = Initiatives.by_reference(reference)
    InitiativeAlerts.create_alert(initiative[0], 'Nueva iniciativa')
    print('Alerts created')


@app.command("long-questions")
def long_questions():
    """Copy the tags of one fully-tagged long question onto the other near-identical ones."""
    from tipi_data.repositories.initiatives import Initiatives

    query = {
        'content.100000': {'$exists': True}
    }
    initiatives = Initiatives.by_query(query)

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
        Initiatives.save(initiative)
        counter += 1
