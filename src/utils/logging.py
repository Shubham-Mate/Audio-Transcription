import logging
import sys


def create_logger():
    logging.basicConfig(
        level=logging.INFO,  # or DEBUG if you want more verbosity
        format="%(asctime)s — %(name)s — %(levelname)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),  # print to console
            # optionally FileHandler / other handlers
        ],
    )
    logger = logging.getLogger()

    return logger
