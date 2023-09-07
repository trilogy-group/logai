import json
import os.path
import logging
import requests
import base64

from attr import dataclass

from logai.config_interfaces import Config
from logai.dataloader.data_loader import DataLoaderConfig
from logai.dataloader.openset_data_loader import get_config


@dataclass
class OpenSearchSetDataLoaderConfig(Config):
    dataset_name: str = None
    index_name: str = None
    host_name: str = None
    port: str = None
    username: str = None
    password: str = None
    query: str = None

class OpenSearchDataLoader():
    def __init__(self, config: OpenSearchSetDataLoaderConfig):
        """
        Initializes opensearch data loader.
        """
        self._dl_config = get_config(config.dataset_name, "")
        self._opensearch_config = config
        self._logger = logging.Logger('opensearchdataloader')
        return

    def load_data(self):
        kwargs = self._dl_config.reader_args
        df = self.retrieve_data()
        for hit in df['hits']['hits']:
            print(hit)
        return ""

    def retrieve_data(self):
        try:
            headers = {
                "Authorization": f"Basic {base64.b64encode(f'{self._opensearch_config.username}:{self._opensearch_config.password}'.encode()).decode()}",
                "Content-Type": "application/json",
            }
            opensearch_url = f"{self._opensearch_config.host_name}/{self._opensearch_config.index_name}/_search"
            if self._opensearch_config.query:
                response = requests.post(opensearch_url,data=self._opensearch_config.query,headers=headers)
            else:
                response = requests.get(opensearch_url, headers=headers)
            if response.status_code == 200:
                data = response.json()
                return data
            else:
                self._logger.error(f"Request failed with status code {response.status_code}: {response.text}")
            return ""
        except Exception as e:
            self._logger.error(f"An error occurred: {e}")

    @property
    def dl_config(self):
        return self._dl_config