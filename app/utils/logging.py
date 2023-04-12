import logging
import logging.handlers
from tqdm import tqdm
import io


class Setting:
    LEVEL         = logging.INFO
    FILENAME      = "outputs/debug.log" 
    MAX_BYTES     = 10 * 1024 * 1024
    BACKUP_COUNT  = 10
    FORMAT        = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    TQDM_FORMAT   = "%(message)s"
    # FORMAT        = "%(asctime)s - %(name)s - %(levelname)s - line %(lineno)s, %(message)s"


def Logger(name, isTqdm=False, filename=None):
    if isTqdm:
        logger          = logging.getLogger(name)
        formatter       = logging.Formatter(Setting.TQDM_FORMAT, datefmt='%Y-%m-%d %H:%M:%S')
        streamHandler   = logging.StreamHandler()
        fileHandler     = logging.handlers.RotatingFileHandler(
            filename    = filename or Setting.FILENAME, 
            maxBytes    = Setting.MAX_BYTES, 
            backupCount = Setting.BACKUP_COUNT)
        
        streamHandler.setFormatter(formatter)
        fileHandler.setFormatter(formatter)

        logger.addHandler(streamHandler)
        logger.addHandler(fileHandler)

        logger.setLevel(Setting.LEVEL)

    else:
        logger          = logging.getLogger(name)
        formatter       = logging.Formatter(Setting.FORMAT, datefmt='%Y-%m-%d %H:%M:%S')
        streamHandler   = logging.StreamHandler()
        fileHandler     = logging.handlers.RotatingFileHandler(
            filename    = filename or Setting.FILENAME, 
            maxBytes    = Setting.MAX_BYTES, 
            backupCount = Setting.BACKUP_COUNT)
        
        streamHandler.setFormatter(formatter)
        fileHandler.setFormatter(formatter)

        logger.addHandler(streamHandler)
        logger.addHandler(fileHandler)

        logger.setLevel(Setting.LEVEL)

    return logger



class TqdmToLogger(io.StringIO):
    
    logger = None
    level = None
    buf = ''

    def __init__(self, logger, level=None):
        super(TqdmToLogger, self).__init__()
        self.logger = logger
        self.level = level or logging.INFO
        self.fileno
    
    def write(self, buf):
        self.buf = buf.strip('\r\n\t')

    def flush(self):
        self.logger.log(self.level, self.buf)



    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg, end=self.terminator)
            self.flush()
        except RecursionError:
            raise
        except Exception:
            self.handleError(record)




