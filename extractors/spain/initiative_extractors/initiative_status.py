import re

from tipi_data.models.initiative import Initiative

from .status_map import STATUS_MAP


UNKNOWN = 'Desconocida'
ON_PROCESS = 'En tramitación'
NOT_FINAL_STATUS = [
        ON_PROCESS,
        UNKNOWN
        ]

def __any_match(regex_list, string):
    for regex in regex_list:
        if re.search(regex, string, re.IGNORECASE):
            return True
    return False

def get_status(history=list(), initiative_type=''):
    if initiative_type == '070':
        return 'Aprobada'
    if not history:
        return UNKNOWN
    for status_map_item in STATUS_MAP:
        if __any_match(status_map_item['latest_history_items'], history[-1]):
            includes = status_map_item['initiative_type']['includes']
            excludes = status_map_item['initiative_type']['excludes']
            if not includes and not excludes:
                return status_map_item['status']
            if includes and initiative_type in includes:
                return status_map_item['status']
            if excludes and initiative_type not in excludes:
                return status_map_item['status']
    return UNKNOWN

def __get_current_status(reference):
    try:
        initiative = Initiative.all.filter(reference=reference).first()
        if 'status' not in initiative:
            return UNKNOWN
        return initiative['status']
    except Exception:
        return UNKNOWN

def has_finished(reference):
    return __get_current_status(reference) not in NOT_FINAL_STATUS

def is_final_status(status):
    return status not in NOT_FINAL_STATUS
