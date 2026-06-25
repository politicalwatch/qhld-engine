from tipi_data.repositories.initiatives import Initiatives


MIMETYPE_FILE_EXTENSIONS = {
        'application/msword': '.doc',
        'application/pdf': '.pdf',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document': '.docx',
        'application/vnd.ms-excel': '.xls',
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': '.xlsx',
        'application/vnd.ms-powerpoint': '.ppt',
        'application/vnd.openxmlformats-officedocument.presentationml.presentation': '.pptx',
        'application/vnd.oasis.opendocument.text': '.odt',
        'application/vnd.oasis.opendocument.spreadsheet': '.ods',
        'application/vnd.oasis.opendocument.presentation': '.odp'
        }

ATTACHMENTS_WORKFLOW = [
        'INICIATIVA',
        'SANCIÓN COMPLETA',
        'LEY'
        ]

def get_current_phase(initiative_id):
    try:
        initiative = Initiatives.get(initiative_id)
        return initiative['extra']['content_reference'], initiative['extra']['content_counter']
    except Exception:
        return '', 0

def get_next_phase(phase_name=''):
    try:
        if phase_name == '':
            return 0, ATTACHMENTS_WORKFLOW[0]
        index = ATTACHMENTS_WORKFLOW.index(phase_name)
        return index+1, ATTACHMENTS_WORKFLOW[index+1]
    except IndexError:
        return -1, ''
