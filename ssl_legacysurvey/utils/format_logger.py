import logging

def create_logger(filename: str = 'logger.log'):
    # create logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # create file handler 
    fh = logging.FileHandler(filename=filename)
    fh.setLevel(logging.DEBUG)

    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    # create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # add formatter to ch
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    # add ch to logger
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger
