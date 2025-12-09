STATUS_MAP = [
        {
            'latest_history_items': [
                r'Boletín Oficial de las Cortes Generales Publicación desde',
                r'Comisión.*desde',
                r'Gobierno Contestación',
                r'Gobierno Reclamación',
                r'Corporación RTVE Contestación',
                r'Junta de Portavoces',
                r'Mesa del Congreso Acuerdo',
                r'Mesa del Congreso Requerimiento',
                r'Mesa del Congreso Calificación',
                r'Mesa del Congreso Reclamación',
                r'Pleno Aprobación desde',
                r'Pleno desde',
                r'Pleno Toma en consideración',
                r'Solicitud de amparo',
                r'Respuesta.*Gobierno',
                r'Senado desde',
                r'Junta Electoral Central desde',
                r'Administración del Estado Contestación',
                r'Entidad Pública Contestación',
                r'Pleno Contestación',
                ],
            'initiative_type': {
                'includes': [],
                'excludes': []
                },
            'status': 'En tramitación'
            },
        {
            'latest_history_items': [
                r'Aprobado con modificaciones',
                r'Aprobado sin modificaciones',
                r'Tramitado con propuesta de resolución',
                r'Concluido',
                r'Tramitado por completo sin',
                ],
            'initiative_type': {
                'includes': [],
                'excludes': ['080', '170', '172', '178', '179', '180', '181', '184', '210', '212', '213', '214', '219']
                },
            'status': 'Aprobada'
            },
        {
                'latest_history_items': [
                    r'Concluido',
                    r'Tramitado por completo sin',
                    ],
                'initiative_type': {
                    'includes': ['170', '172', '178', '179', '180', '181', '184'],
                    'excludes': []
                    },
                'status': 'Respondida'
                },
        {
                'latest_history_items': [
                    r'Tramitado por completo sin',
                    ],
                'initiative_type': {
                    'includes': ['210', '212', '213', '214', '219'],
                    'excludes': [],
                    },
                'status': 'Celebrada'
                },
        {
                'latest_history_items': [
                    r'Convalidado',
                    ],
                'initiative_type': {
                    'includes': [],
                    'excludes': [],
                    },
                'status': 'Convalidada'
                },
        {
                'latest_history_items': [
                    r'Convertido',
                    ],
                'initiative_type': {
                    'includes': [],
                    'excludes': []
                    },
                'status': 'Convertida en otra'
                },
        {
                'latest_history_items': [
                    r'Subsumido en otra iniciativa',
                    ],
                'initiative_type': {
                    'includes': [],
                    'excludes': []
                    },
                'status': 'Acumulada en otra'
                },
        {
                'latest_history_items': [
                    r'Inadmitido a trámite',
                    ],
                'initiative_type': {
                    'includes': [],
                    'excludes': []
                    },
                'status': 'No admitida a trámite'
                },
        {
                'latest_history_items': [
                    r'Decaído',
                    ],
                'initiative_type': {
                    'includes': [],
                    'excludes': []
                    },
                'status': 'No debatida'
                },
        {
                'latest_history_items': [
                    r'Rechazado',
                    ],
                'initiative_type': {
                    'includes': [],
                    'excludes': []
                    },
                'status': 'Rechazada'
                },
        {
                'latest_history_items': [
                    r'Tramitado por completo sin',
                    ],
                'initiative_type': {
                    'includes': ['080'],
                    'excludes': []
                    },
                'status': 'Rechazada'
                },
        {
                'latest_history_items': [
                    r'Retirado',
                    ],
                'initiative_type': {
                    'includes': [],
                    'excludes': []
                    },
                'status': 'Retirada'
                },
        {
                'latest_history_items': [
                    r'No celebración',
                    ],
                'initiative_type': {
                    'includes': [],
                    'excludes': []
                    },
                'status': 'No celebrada'
                },
        {
                'latest_history_items': [
                    r'Derogado',
                    ],
                'initiative_type': {
                    'includes': [],
                    'excludes': []
                    },
                'status': 'Derogada'
                },
        {
                'latest_history_items': [
                    r'Extinguido',
                    ],
                'initiative_type': {
                    'includes': [],
                    'excludes': []
                    },
                'status': 'Extinguida'
                },
        {
                'latest_history_items': [
                    r'Caducado',
                    ],
                'initiative_type': {
                    'includes': [],
                    'excludes': []
                    },
                'status': 'Caducada'
                },
        ]
