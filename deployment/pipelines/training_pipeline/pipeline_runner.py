import sys

import pandas as pd
import numpy as np

from pyspark.sql import SparkSession

from lab_project_demo.config.ConfigProvider import read_config, setupMlflowConf
from lab_project_demo.data.TrainingPipeline import TrainingPipeline

spark = SparkSession.builder.appName('ForecastingTest').getOrCreate()
conf = read_config('train_config.yaml', sys.argv[1])
setupMlflowConf(conf)
p = TrainingPipeline(spark, conf['data-path'], conf['model-name'])
p.run()