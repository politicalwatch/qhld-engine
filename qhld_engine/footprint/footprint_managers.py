from datetime import timedelta

# --- Status ----------------------------------------------------------------

EXCLUDED_STATUS = ['No admitida a trámite', 'Retirada']
APPROVED_STATUS = 'Aprobada'

# --- Initiative-type weights ----------------------------------------------

TYPES_FOUR = [
    'Comparecencia del Gobierno en Comisión (art. 44)',
    'Comparecencia del Gobierno en Comisión (arts. 202 y 203)',
    'Comparecencia de autoridades y funcionarios en Comisión',
    'Comparec. autoridades y funcionarios en Com. Mx. solicitada en Senado',
    'Otras comparecencias en Comisión',
    'Interpelación urgente',
    'Interpelación ordinaria',
    'Solicitud de informe a la Administración del Estado (art. 7)',
    'Solicitud de informe a otra Entidad Pública (art. 7)',
    'Solicitud de informe a la Administración del Estado (art. 44)',
    'Solicitud de informe a otra Entidad Pública (art. 44)',
    'Otras solicitudes de informe (art. 44)',
]

TYPES_TEN = [
    'Pregunta oral en Pleno',
    'Pregunta oral al Gobierno en Comisión',
    'Moción de censura',
    'Pregunta oral a la Corporación RTVE',
]

TYPES_FOURTY = [
    'Proposición no de Ley ante el Pleno',
    'Proposición no de Ley en Comisión',
    'Moción consecuencia de interpelación ordinaria',
    'Moción consecuencia de interpelación urgente',
]

TYPES_EIGHTY = [
    'Proposición de ley de Grupos Parlamentarios del Congreso',
    'Proposición de ley de Diputados',
    'Proyecto de reforma Constitucional',
    'Proposición de reforma Constitucional de Grupos Parlamentarios',
    'Proposición de reforma constitucional de Comunidades Autónomas',
]

# ``(weight, types)`` pairs, ordered high-to-low (order is irrelevant to the sum but
# keeps the $switch readable).
BASE_WEIGHTS = [
    (80, TYPES_EIGHTY),
    (40, TYPES_FOURTY),
    (10, TYPES_TEN),
    (4, TYPES_FOUR),
]

# Extra weight granted only when the initiative was approved.
TYPES_APPROVED_SIXTY = [
    'Proposición de ley de Grupos Parlamentarios del Congreso',
    'Proposición de ley de Diputados',
]
TYPES_APPROVED_TWENTY = [
    'Proposición no de Ley ante el Pleno',
    'Proposición no de Ley en Comisión',
]
APPROVED_BONUS = [
    (60, TYPES_APPROVED_SIXTY),
    (20, TYPES_APPROVED_TWENTY),
]

AUTHOR_FIELD = {
    'deputy': 'author_deputies',
    'parliamentarygroup': 'author_parliamentarygroups',
}


def weight_of(initiative_type_alt, status):
    """Pure-Python weight of a single initiative, used by tests and as the reference
    for the Mongo expression below."""
    weight = 0
    for value, types in BASE_WEIGHTS:
        if initiative_type_alt in types:
            weight += value
            break
    if status == APPROVED_STATUS:
        for value, types in APPROVED_BONUS:
            if initiative_type_alt in types:
                weight += value
    return weight


def _weight_expr():
    """Mongo expression mirroring :func:`weight_of` for use inside a pipeline."""
    base = {
        '$switch': {
            'branches': [
                {'case': {'$in': ['$initiative_type_alt', types]}, 'then': value}
                for value, types in BASE_WEIGHTS
            ],
            'default': 0,
        }
    }
    bonus = {
        '$cond': {
            'if': {'$eq': ['$status', APPROVED_STATUS]},
            'then': {
                '$switch': {
                    'branches': [
                        {'case': {'$in': ['$initiative_type_alt', types]}, 'then': value}
                        for value, types in APPROVED_BONUS
                    ],
                    'default': 0,
                }
            },
            'else': 0,
        }
    }
    return {'$add': [base, bonus]}


def _valid_status_match():
    return {'status': {'$not': {'$in': EXCLUDED_STATUS}}}


# --- Pipeline builders -----------------------------------------------------

def topic_scores_pipeline(typeof):
    """One row per (author, topic): the weighted, per-topic score.

    ``score = Σ weight * (topic_alignment.percentage / 100) / num_deputies`` over the
    entity's valid initiatives. For groups there is no per-coauthor division
    (``num_deputies`` is fixed to 1, as in the original algorithm)."""
    author = AUTHOR_FIELD[typeof]
    num_deputies = {'$size': {'$ifNull': ['$author_deputies', []]}} \
        if typeof == 'deputy' else 1
    return [
        {'$match': _valid_status_match()},
        {'$addFields': {'_w': _weight_expr()}},
        {'$match': {'_w': {'$gt': 0}}},
        {'$addFields': {'_nd': num_deputies}},
        {'$unwind': '$tagged'},
        {'$unwind': '$tagged.topic_alignment'},
        {'$unwind': f'${author}'},
        {'$addFields': {
            '_contrib': {
                '$cond': {
                    'if': {'$gt': ['$_nd', 0]},
                    'then': {
                        '$divide': [
                            {'$multiply': [
                                '$_w',
                                {'$divide': ['$tagged.topic_alignment.percentage', 100]},
                            ]},
                            '$_nd',
                        ]
                    },
                    'else': 0,
                }
            }
        }},
        {'$group': {
            '_id': {'e': f'${author}', 't': '$tagged.topic_alignment.topic'},
            'score': {'$sum': '$_contrib'},
        }},
    ]


def global_scores_pipeline(typeof):
    """One row per author: the global (topic-agnostic) score and the last valid
    creation date (for the inactivity penalty).

    The global score counts each initiative once per matching weight tier, without
    the topic-alignment or per-coauthor factors — i.e. ``Σ weight``. Only scorable
    initiatives (weight > 0) are considered: they define the score *and* ``last``, so
    the inactivity penalty measures staleness of *scorable* activity only. Filtering
    weight-0 documents leaves the ``score`` sum unchanged (they add 0)."""
    author = AUTHOR_FIELD[typeof]
    return [
        {'$match': _valid_status_match()},
        {'$addFields': {'_w': _weight_expr()}},
        {'$match': {'_w': {'$gt': 0}}},
        {'$unwind': f'${author}'},
        {'$group': {
            '_id': f'${author}',
            'score': {'$sum': '$_w'},
            'last': {'$max': '$created'},
        }},
    ]


def topic_last_dates_pipeline(typeof):
    """One row per (author, topic): the last valid creation date, restricted to
    scorable initiatives (weight > 0), for the per-topic inactivity penalty."""
    author = AUTHOR_FIELD[typeof]
    return [
        {'$match': _valid_status_match()},
        {'$addFields': {'_w': _weight_expr()}},
        {'$match': {'_w': {'$gt': 0}}},
        {'$unwind': f'${author}'},
        {'$unwind': '$tagged'},
        {'$unwind': '$tagged.topics'},
        {'$group': {
            '_id': {'e': f'${author}', 't': '$tagged.topics'},
            'last': {'$max': '$created'},
        }},
    ]


# --- Inactivity penalty ----------------------------------------------------

DAYS_IN_MONTH = 30


def inactivity_penalty(last_date, today):
    """Fraction (0..0.5) to subtract from a score based on how stale the entity's last
    *scorable* initiative is (``last_date`` is fed only from weight > 0 initiatives).
    Months approximated as 30 days, as in the original."""
    if not last_date:
        return 0
    if last_date <= today - timedelta(days=DAYS_IN_MONTH * 12):
        return 0.50
    if last_date <= today - timedelta(days=DAYS_IN_MONTH * 6):
        return 0.25
    if last_date <= today - timedelta(days=DAYS_IN_MONTH * 3):
        return 0.10
    return 0


# --- Deputy contact bonus --------------------------------------------------

class FootprintDeputyManager:
    def __init__(self, deputy):
        self.deputy = deputy

    def __exists(self, field):
        return field in self.deputy

    def __not_empty(self, field):
        return self.deputy[field] != ''

    def __has_field(self, field):
        return self.__exists(field) and self.__not_empty(field)

    def has_email(self):
        return self.__has_field('email')

    def has_social(self):
        return self.__has_field('twitter') or self.__has_field('facebook')

    def compute_email(self):
        if self.has_email():
            return 40
        return 0

    def compute_social(self):
        if self.has_social():
            return 40
        return 0
