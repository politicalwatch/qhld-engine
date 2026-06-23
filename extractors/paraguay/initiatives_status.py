import re

from tipi_data.repositories.initiatives import Initiatives


def is_final_state(status):
    FINAL_STATES_REGEX = [
            'archivado',
            'publicado',
            'retirado',
            'contestado',
            'suspendido'
            ]
    control = False
    for regex in FINAL_STATES_REGEX:
        if re.search(regex, status, re.IGNORECASE):
            control = True
            break
    return control

def has_same_saved_status(initiative):
    saved_status = get_current_status(str(initiative['idProyecto']))
    new_status = initiative['estadoProyecto']
    return saved_status == new_status

def get_current_status(initiative_id):
    try:
        initiative = Initiatives.get(initiative_id)
        return initiative['status']
    except Exception:
        return ''

def has_finished(initiative):
    return is_final_state(initiative['estadoProyecto']) and has_same_saved_status(initiative)
