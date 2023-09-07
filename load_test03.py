import os
from logai.applications.log_anomaly_detection import LogAnomalyDetection
from logai.applications.application_interfaces import WorkFlowConfig
import json

json_config = """{
      "opensearch_set_data_loader_config": {
        "dataset_name": "opensearch",
        "index_name": "central_eks-2023.09.06",
        "host_name": "https://kibana.private.central-eks.aureacentral.com",
        "port": "443",
        "username": "central",
        "password": "we<3kube"
      },
      "preprocessor_config": {
          "custom_delimiters_regex":[]
      },
      "log_parser_config": {
        "parsing_algorithm": "drain",
        "parsing_algo_params": {
          "sim_th": 0.5,
          "depth": 5
        }
      },
      "feature_extractor_config": {
          "group_by_category": ["Level"],
          "group_by_time": "1s"
      },
      "log_vectorizer_config": {
          "algo_name": "word2vec"
      },
      "categorical_encoder_config": {
          "name": "label_encoder"
      },
      "anomaly_detection_config": {
          "algo_name": "one_class_svm"
      }
}
"""

config = json.loads(json_config)
workflow_config = WorkFlowConfig.from_dict(config)
app = LogAnomalyDetection(workflow_config)
app.execute()
test1 = app.anomaly_results.head(5)
print(test1)
