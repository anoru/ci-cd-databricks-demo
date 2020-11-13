import sys

import pandas as pd
import numpy as np

from pyspark.sql import SparkSession

from lab_project_demo.config.ConfigProvider import read_config, setupMlflowConf
from lab_project_demo.models.ModelEvaluationPipeline import ModelEvaluationPipeline

spark = SparkSession.builder.appName('ForecastingTest').getOrCreate()
conf = read_config('train_config.yaml', sys.argv[1])
experimentID = setupMlflowConf(conf)
p = ModelEvaluationPipeline(spark, experimentID, conf['model-name'], conf['data-path'] )
p.run()