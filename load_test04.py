import os
from logai.dataloader.opensearch_data_loader import OpenSearchSetDataLoaderConfig, OpenSearchDataLoader
from logai.preprocess.preprocessor import PreprocessorConfig, Preprocessor
from logai.utils import constants
from logai.information_extraction.log_parser import LogParser, LogParserConfig
from logai.algorithms.parsing_algo.drain import DrainParams
from logai.information_extraction.log_vectorizer import VectorizerConfig, LogVectorizer
from logai.information_extraction.categorical_encoder import CategoricalEncoderConfig, CategoricalEncoder
from logai.information_extraction.feature_extractor import FeatureExtractorConfig, FeatureExtractor
from logai.algorithms.clustering_algo.kmeans import KMeansParams
from logai.analysis.clustering import ClusteringConfig, Clustering


#File Configuration


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
            "query":
            {
                "bool":
                {
                    "must":
                    [
                        {
                            "match":
                            {
                                "kubernetes.namespace_name": "cdb-dev"
                            }
                        },
                        {
                            "range":
                            {
                                "@timestamp":
                                {
                                    "gte": "2023-09-07T22:00:00.000",
                                    "lte" : "2023-09-07T22:00:30.000"
                                }
                            }
                        }
                    ]
                }
            }
        }
        """)
)

logrecord = data_loader.load_data()

#logrecord.to_dataframe().head(5)
loglines = logrecord.body[constants.LOGLINE_NAME]
attributes = logrecord.attributes
preprocessor_config = PreprocessorConfig(
    custom_replace_list=[
        [r"(?<=blk_)[-\d]+", "<block_id>"],
        [r"\d+\.\d+\.\d+\.\d+", "<IP>"],
        [r"(/[-\w]+)+", "<file_path>"],
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

clustering_config = ClusteringConfig(
    algo_name='kmeans',
    algo_params=KMeansParams(
        n_clusters=7
    )
)

log_clustering = Clustering(clustering_config)

log_clustering.fit(feature_vector)

cluster_id = log_clustering.predict(feature_vector).astype(str).rename('cluster_id')

dump1 = logrecord.to_dataframe().join(cluster_id).head(5)
print(dump1)