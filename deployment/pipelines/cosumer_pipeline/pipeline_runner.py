import sys

from pyspark.sql import SparkSession
import os
from lab_project_demo.models.ConsumerPipeline import ConsumerPipeline
from lab_project_demo.config.ConfigProvider import read_config, setupMlflowConf

spark = SparkSession.builder.appName('Test').getOrCreate()
conf = read_config('consumer_config.yaml', sys.argv[1])
setupMlflowConf(conf)

p = ConsumerPipeline(spark, conf['data-path'],conf['output-path'],conf['model-name'], conf['stage'])
p.run()

spark.read.load(conf['output-path']).show(1000, False)