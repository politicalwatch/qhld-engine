import logging

from qhld_engine.infrastructure.config.settings import get_settings


def get_logger(name):
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        '%(asctime)s : %(levelname)s : %(name)s : %(message)s'
        ))
    log = logging.getLogger(name)
    log.setLevel(get_settings().loglevel)
    log.addHandler(handler)
    return log
