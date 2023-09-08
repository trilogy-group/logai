import os

from logai.dataloader.opensearch_data_loader import OpenSearchSetDataLoaderConfig, OpenSearchDataLoader
from logai.preprocess.preprocessor import PreprocessorConfig, Preprocessor
from logai.information_extraction.log_parser import LogParser, LogParserConfig
from logai.algorithms.parsing_algo.drain import DrainParams
from logai.utils import constants
from logai.information_extraction.feature_extractor import FeatureExtractorConfig, FeatureExtractor
from logai.analysis.anomaly_detector import AnomalyDetector, AnomalyDetectionConfig
from sklearn.model_selection import train_test_split
import pandas as pd
from logai.information_extraction.log_vectorizer import VectorizerConfig, LogVectorizer
from logai.information_extraction.categorical_encoder import CategoricalEncoderConfig, CategoricalEncoder
from logai.information_extraction.feature_extractor import FeatureExtractorConfig, FeatureExtractor
from logai.algorithms.anomaly_detection_algo.isolation_forest import IsolationForestParams
from logai.analysis.anomaly_detector import AnomalyDetectionConfig, AnomalyDetector

dataset_name = "opensearch"
data_loader = OpenSearchDataLoader(
    OpenSearchSetDataLoaderConfig(
        dataset_name=dataset_name,
        index_name='central_eks-2023.09.07',
        host_name='https://kibana.private.central-eks.aureacentral.com',
        port="443",
        username='central',
        password='we<3kube',
        query="""
        {
            "query": {
                "bool": {
        	        "filter":{
				        "range":{
                            "@timestamp":{
                                "gte": "2023-09-07T22:00:00.000",
                                "lte" : "2023-09-07T22:20:00.000"
                            }
                        }
        	        },
        	        "should": [
        		        {
        			        "match": {
        				        "kubernetes.namespace_name": "cdb-dev"
        			        }
        		        },
        		        {
        			        "match": {
        				        "kubernetes.namespace_name": "cdb-prod"
        			        }
        		        },
        		        {
        			        "match": {
        				        "kubernetes.namespace_name": "placeable-prod"
        			        }
        		        }
        	        ]
                }
            }
        }
        """)
)
logrecord = data_loader.load_data()

logrecord.to_dataframe().head(5)

loglines = logrecord.body[constants.LOGLINE_NAME]
attributes = logrecord.attributes
preprocessor_config = PreprocessorConfig(
    custom_replace_list=[
        [r"\d+\.\d+\.\d+\.\d+", "<IP>"],   # retrieve all IP addresses and replace with <IP> tag in the original string.
    ]
)

preprocessor = Preprocessor(preprocessor_config)

clean_logs, custom_patterns = preprocessor.clean_log(
    loglines
)

# parsing
parsing_algo_params = DrainParams(
    sim_th=0.5, depth=5
)

log_parser_config = LogParserConfig(
    parsing_algorithm="drain",
    parsing_algo_params=parsing_algo_params
)

parser = LogParser(log_parser_config)
parsed_result = parser.parse(clean_logs)

parsed_loglines = parsed_result['parsed_logline']

config = FeatureExtractorConfig(
    group_by_time="15min",
    group_by_category=['parsed_logline', 'PodName', 'ContainerName'],
)

feature_extractor = FeatureExtractor(config)

timestamps = logrecord.timestamp['timestamp']
parsed_loglines = parsed_result['parsed_logline']
counter_vector = feature_extractor.convert_to_counter_vector(
    log_pattern=parsed_loglines,
    attributes=attributes,
    timestamps=timestamps
)

counter_vector.head(5)
#----------------------------------------------------------------------------
counter_vector["attribute"] = counter_vector.drop(
                [
                    constants.LOG_COUNTS,
                    constants.LOG_TIMESTAMPS,
                    constants.EVENT_INDEX
                ],
                axis=1
            ).apply(
                lambda x: "-".join(x.astype(str)), axis=1
            )

attr_list = counter_vector["attribute"].unique()

anomaly_detection_config = AnomalyDetectionConfig(
    algo_name='dbl'
)

res = pd.DataFrame()
for attr in attr_list:
    temp_df = counter_vector[counter_vector["attribute"] == attr]
    if temp_df.shape[0] >= constants.MIN_TS_LENGTH:
        train, test = train_test_split(
            temp_df[[constants.LOG_TIMESTAMPS, constants.LOG_COUNTS]],
            shuffle=False,
            train_size=0.3
        )
        anomaly_detector = AnomalyDetector(anomaly_detection_config)
        anomaly_detector.fit(train)
        anom_score = anomaly_detector.predict(test)
        res = res.append(anom_score)

# Get anomalous datapoints
anomalies = counter_vector.iloc[res[res>0].index]
print(anomalies.head(5))

#-----------------------------------------------------------------------------------------


vectorizer_config = VectorizerConfig(
    algo_name = "word2vec"
)

vectorizer = LogVectorizer(
    vectorizer_config
)

# Train vectorizer
vectorizer.fit(parsed_loglines)

# Transform the loglines into features
log_vectors = vectorizer.transform(parsed_loglines)


encoder_config = CategoricalEncoderConfig(name="label_encoder")

encoder = CategoricalEncoder(encoder_config)

attributes_encoded = encoder.fit_transform(attributes)

timestamps = logrecord.timestamp['timestamp']

config = FeatureExtractorConfig(
    max_feature_len=100
)

feature_extractor = FeatureExtractor(config)

_, feature_vector = feature_extractor.convert_to_feature_vector(log_vectors, attributes_encoded, timestamps)

from sklearn.model_selection import train_test_split

train, test = train_test_split(feature_vector, train_size=0.7, test_size=0.3)


algo_params = IsolationForestParams(
    n_estimators=10,
    max_features=100
)
config = AnomalyDetectionConfig(
    algo_name='isolation_forest',
    algo_params=algo_params
)

anomaly_detector = AnomalyDetector(config)
anomaly_detector.fit(train)
res = anomaly_detector.predict(test)
# obtain the anomalous datapoints
anomalies = res[res==1]

print(loglines.iloc[anomalies.index].head(5))
print(attributes.iloc[anomalies.index].head(5))

print('_______________________________________________')