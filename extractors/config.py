from os import environ as env


MODULE_EXTRACTOR = env.get('MODULE_EXTRACTOR', 'spain')
ID_LEGISLATURA = int(env.get('ID_LEGISLATURA', '0'))
CURRENT_LEGISLATURE = bool(env.get('CURRENT_LEGISLATURE', True))
LIMIT_DATE_TO_SYNC = env.get('LIMIT_DATE_TO_SYNC', '2000-01-01')
REDIS_HOST = env.get('REDIS_HOST', 'localhost')
REDIS_PORT = int(env.get('REDIS_PORT', '6379'))
REDIS_DB_CHECK = int(env.get('REDIS_DB_CHECK', '0'))
REDIS_DB_DENYLIST = int(env.get('REDIS_DB_DENYLIST', '1'))

AMENDMENTS_FEATURE = bool(env.get('AMENDMENTS_FEATURE', False))
