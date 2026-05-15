import os
import sys
import logging

logging_str = "[%(asctime)s: %(levelname)s: %(module)s: %(message)s]"

handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
log_filepath = os.getenv("SAFE_RESPONSE_LOG_FILE", "").strip()
if log_filepath:
    log_dir = os.path.dirname(log_filepath)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    handlers.append(logging.FileHandler(log_filepath))

logging.basicConfig(
    level=logging.INFO,
    format=logging_str,
    handlers=handlers,
)

logger = logging.getLogger("SafeResponseEngine")
