import os
from datetime import datetime, timedelta
import pytz
from logai.dataloader.grafanaloki_data_loader import GrafanaLokiDataLoader,GrafanaLokiDataLoaderConfig

end_time = datetime.now(pytz.UTC)
start_time = end_time - timedelta(minutes=5)


dataset_name = "grafanaloki"
data_loader = GrafanaLokiDataLoader(
        GrafanaLokiDataLoaderConfig(
        dataset_name=dataset_name,
        host_name='https://logcentral.private.central-eks.aureacentral.com',
        port="443",
        username='logcentral',
        password='eB2rNGqdvF',
        org_id='central-eks',
        query='{namespace="placeable-prod"}',
        start_time=start_time,
        end_time=end_time,
    )
)
#query='{namespace="placeable-prod"} |= ""',

logrecord = data_loader.load_data()
logrecord.to_dataframe().head(5)