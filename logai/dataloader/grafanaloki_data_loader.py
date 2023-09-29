import json
import os.path
import logging
from typing import Any
import requests
import base64
import re
from datetime import datetime, timedelta

from attr import dataclass
import pandas as pd
from logai.utils import constants
from logai.config_interfaces import Config
#from logai.dataloader.data_loader import DataLoaderConfig
from logai.dataloader.openset_data_loader import get_config
from logai.dataloader.data_model import LogRecordObject




@dataclass
class GrafanaLokiDataLoaderConfig(Config):
    dataset_name: str = None
    org_id: str = None
    search_labels: dict = None
    host_name: str = None
    port: str = None
    username: str = None
    password: str = None
    query: str = None
    start_time: datetime = None
    end_time: datetime = None

class GrafanaLokiDataLoader():
    def __init__(self, config: GrafanaLokiDataLoaderConfig):
        self.dl_config = get_config(config.dataset_name, "")
        self._default_page_size = 200
        self._grafanaloki_config = config
        self._logger = logging.getLogger('grafanalokidataloader')
        return

    def load_data(self):
        kwargs = self.dl_config.reader_args
        if self._verify_connectivity():
            df = self.retrieve_data()
        return self._create_log_record_object("")

    def retrieve_data(self):
        results = []
        lapses = self._get_timestamp_ranges(self._grafanaloki_config.start_time, self._grafanaloki_config.end_time)
        for lapse in lapses:
            resp_data = self._get_data_from_range(lapse[0], lapse[1], 5000)
            results.append(resp_data)
        return results

    def _get_data_from_range(self, start_time, end_time, limit):
        headers = self._get_auth_headers()
        request_params = {
            "query": self._grafanaloki_config.query,
            "start": int(start_time.timestamp()),
            "end": int(end_time.timestamp()),
            "limit": limit,
        }
        resp_data = requests.get(f"{self._grafanaloki_config.host_name}:{self._grafanaloki_config.port}/loki/api/v1/query_range", headers=headers,params=request_params)
        if resp_data.status_code != 200:
            self._logger.error(f"Error getting the data {self._grafanaloki_config.host_name}:{self._grafanaloki_config.port} - status code {resp_data.status_code}")
            return None
        return resp_data.json()


    def _get_timestamp_ranges(self,start_time, end_time):
        interval = timedelta(seconds=5)
        time_intervals = []
        current_time = start_time
        while current_time < end_time:
            next_time = current_time + interval
            time_intervals.append((current_time, next_time))
            current_time = next_time
        return time_intervals

    def _create_log_record_object(self, df: pd.DataFrame):
        log_record = LogRecordObject()
        return log_record

    def _get_labels(self):
        headers = self._get_auth_headers()
        resp_labels = requests.get(f"{self._grafanaloki_config.host_name}:{self._grafanaloki_config.port}/loki/api/v1/labels", headers=headers)
        if resp_labels.status_code != 200:
            self._logger.error(f"Error getting the labels {self._grafanaloki_config.host_name}:{self._grafanaloki_config.port} - status code {resp_labels.status_code}")
            return {}
        return resp_labels.json()

    def _verify_connectivity(self):
        headers = self._get_auth_headers()
        response = requests.get(f"{self._grafanaloki_config.host_name}:{self._grafanaloki_config.port}/loki/api/v1/status/buildinfo", headers=headers)
        if response.status_code == 200:
            return True
        else:
            self._logger.error(f"Error connecting to Grafana Loki at {self._grafanaloki_config.host_name}:{self._grafanaloki_config.port} - status code {response.status_code}")
            return False

    def _get_auth_headers(self):
        headers = {
            "Authorization": f"Basic {base64.b64encode(f'{self._grafanaloki_config.username}:{self._grafanaloki_config.password}'.encode()).decode()}",
            "X-Scope-OrgID": self._grafanaloki_config.org_id,
            "Content-Type": "application/json",
        }
        return headers