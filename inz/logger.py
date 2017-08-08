import logging


def get_level(level: str):
    level = level.upper()
    return getattr(logging, level)


def create_logger(name: str = None, level='INFO', filename=None):
    level = get_level(level)

    logging.basicConfig(filename=filename, level=level)

    logger = logging.getLogger(name)
    formatter = logging.Formatter('%(asctime)s|%(name)s|%(levelname)s|%(message)s')

    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    logger.debug('New logger created')
    return logger


if __name__ == '__main__':
    logger = create_logger(level='DEBUG', filename='somefile.log')
    logger.debug('hello world')
