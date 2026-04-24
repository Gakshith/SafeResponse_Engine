import ssl
from urllib import request
from urllib.parse import urlparse
from src.saferesponse_engine.entity.config_entity import UserQueryConfig
from src.saferesponse_engine import logger
from src.saferesponse_engine.utils.common import get_size

class UserQuery:
    def __init__(self, config: UserQueryConfig):
        self.config = config

    @staticmethod
    def _resolve_source_url(source_url: str) -> str:
        parsed = urlparse(source_url)
        if parsed.netloc == "github.com" and "/blob/" in parsed.path:
            owner, repo, _, branch, *file_parts = parsed.path.strip("/").split("/")
            return f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{'/'.join(file_parts)}"
        return source_url

    def download_file(self):
         if not self.config.local_data_file.exists():
              self.config.local_data_file.parent.mkdir(parents=True, exist_ok=True)
              ctx = ssl.create_default_context()
              source_url = self._resolve_source_url(self.config.source_url)
              with request.urlopen(source_url, context=ctx) as r:
                 text = r.read().decode("utf-8")
              self.config.local_data_file.write_text(text, encoding="utf-8")
              logger.info(f"Downloaded file to: {self.config.local_data_file}")
         else:
             logger.info(f"File already exists of size: {get_size(self.config.local_data_file)}")
