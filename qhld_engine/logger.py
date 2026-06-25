import logging
import os


def get_logger(name):
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        '%(asctime)s : %(levelname)s : %(name)s : %(message)s'
        ))
    log = logging.getLogger(name)
    log.setLevel(os.environ.get("LOGLEVEL", "INFO"))
    log.addHandler(handler)
    return log
