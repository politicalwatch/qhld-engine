from os import environ as env


USE_ALERTS = env.get('USE_ALERTS', 'False') == 'True'
REASONS = {
        'new': 'Nueva iniciativa',
        'published': 'Texto publicado',
        'finished': 'Tramitación finalizada'
        }
