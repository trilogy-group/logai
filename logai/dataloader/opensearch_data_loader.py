import json
import os.path
import logging
import requests
import base64
import re

from attr import dataclass
import pandas as pd
from logai.utils import constants

from logai.config_interfaces import Config
from logai.dataloader.data_loader import DataLoaderConfig
from logai.dataloader.openset_data_loader import get_config
from logai.dataloader.data_model import LogRecordObject


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
        self._default_page_size = 200
        self._opensearch_config = config
        self._logger = logging.Logger('opensearchdataloader')
        return

    def load_data(self):
        kwargs = self._dl_config.reader_args
        df = self.retrieve_data()
        logdf = self._read_from_opensearch(df)
        return self._create_log_record_object(logdf)

    def _read_from_opensearch(self, df):
        log_messages = []
        headers, log_regex = self.get_headers_regex(self._dl_config.reader_args['log_format'])
        body_field = self._dl_config.dimensions['body'][0]
        for row in df:
            try:
                match = log_regex.search(row['_source'][body_field].strip())
                message = [match.group(header) for header in headers]
                log_messages.append(message)
            except Exception as e:
                logging.error('Opensearch row read failed. Exception {}.'.format(e))
        logdf = pd.DataFrame(log_messages, columns=headers, dtype=str)
        return logdf


    def get_headers_regex(self, log_format):
        headers = []
        splitters = re.split(r"(<[^<>]+>)", log_format)
        regex = ""
        for k in range(len(splitters)):
            if k % 2 == 0:
                splitter = re.sub(" +", "\\\s+", splitters[k])
                regex += splitter
            else:
                header = splitters[k].strip("<").strip(">")
                regex += "(?P<%s>.*?)" % header
                headers.append(header)
        regex = re.compile("^" + regex + "$")
        return headers, regex

    def retrieve_data(self):
        results = []
        try:
            headers = {
                "Authorization": f"Basic {base64.b64encode(f'{self._opensearch_config.username}:{self._opensearch_config.password}'.encode()).decode()}",
                "Content-Type": "application/json",
            }
            total_size = 0
            pending_to_retrieve = self._default_page_size
            current_page = 0
            while pending_to_retrieve > 0:
                opensearch_url = f"{self._opensearch_config.host_name}/{self._opensearch_config.index_name}/_search?size={self._default_page_size}&from={current_page}"
                if self._opensearch_config.query:
                    response = requests.post(opensearch_url,data=self._opensearch_config.query,headers=headers)
                else:
                    response = requests.get(opensearch_url, headers=headers)
                if response.status_code == 200:
                    data = response.json()
                    if current_page == 0:
                        total_size = data['hits']['total']['value']
                        pending_to_retrieve = total_size
                        logging.info(f"Total to retrieve: {total_size}")
                    current_page += self._default_page_size
                    pending_to_retrieve = pending_to_retrieve - self._default_page_size
                    logging.info(f"Pending to retrieve: {pending_to_retrieve}")
                    results.extend(data['hits']['hits'])
                else:
                    self._logger.error(f"Request failed with status code {response.status_code}: {response.text}")
        except Exception as e:
            self._logger.error(f"An error occurred: {e}")
        return results

    @property
    def dl_config(self):
        return self._dl_config
    
    def _create_log_record_object(self, df: pd.DataFrame):
        dims = self._dl_config.dimensions
        log_record = LogRecordObject()
        if not dims:
            selected = pd.DataFrame(
                df.agg(lambda x: " ".join(x.values), axis=1).rename(
                    constants.LOGLINE_NAME
                )
            )
            setattr(log_record, "body", selected)
        else:
            for field in LogRecordObject.__dataclass_fields__:
                if field in dims.keys():
                    selected = df[list(dims[field])]
                    print(field)
                    if field == "timestamp":
                        if len(selected.columns) > 1:
                            selected = pd.DataFrame(
                                selected.agg(
                                    lambda x: " ".join(x.values), axis=1
                                ).rename(constants.LOG_TIMESTAMPS)
                            )
                        selected.columns = [constants.LOG_TIMESTAMPS]
                        if self._dl_config.infer_datetime and self._dl_config.datetime_format:
                            datetime_format = self._dl_config.datetime_format
                            selected[constants.LOG_TIMESTAMPS] = pd.to_datetime(
                                selected[constants.LOG_TIMESTAMPS],
                                format=datetime_format,
                            )
                    setattr(log_record, field, selected)
        return log_record