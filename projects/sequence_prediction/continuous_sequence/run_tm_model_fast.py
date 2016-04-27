## ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2013-2015, Numenta, Inc.  Unless you have an agreement
# with Numenta, Inc., for a separate license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Affero Public License for more details.
#
# You should have received a copy of the GNU Affero Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------


import importlib
import time

from optparse import OptionParser
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import rcParams

rcParams.update({'figure.autolayout': True})

from nupic.frameworks.opf.metrics import MetricSpec
from nupic.frameworks.opf.modelfactory import ModelFactory
from nupic.frameworks.opf.predictionmetricsmanager import MetricsManager
from nupic.frameworks.opf import metrics
from htmresearch.frameworks.opf.clamodel_custom import CLAModel_custom
import nupic_output


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from errorMetrics import *
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})


plt.ion()
DESCRIPTION = (
  "Starts a NuPIC model from the model params returned by the swarm\n"
  "and pushes each line of input from the gym into the model. Results\n"
  "are written to an output file (default) or plotted dynamically if\n"
  "the --plot option is specified.\n"
  "NOTE: You must run ./swarm.py before this, because model parameters\n"
  "are required to run NuPIC.\n"
)


DATA_DIR = "./data"
MODEL_PARAMS_DIR = "./model_params"


def getMetricSpecs(predictedField, stepsAhead=5):
  _METRIC_SPECS = (
      MetricSpec(field=predictedField, metric='multiStep',
                 inferenceElement='multiStepBestPredictions',
                 params={'errorMetric': 'negativeLogLikelihood',
                         'window': 1000, 'steps': stepsAhead}),
      MetricSpec(field=predictedField, metric='multiStep',
                 inferenceElement='multiStepBestPredictions',
                 params={'errorMetric': 'nrmse', 'window': 1000,
                         'steps': stepsAhead}),
  )
  return _METRIC_SPECS


def createModel(modelParams):
  model = ModelFactory.create(modelParams)
  model.enableInference({"predictedField": predictedField})
  return model


def getModelParamsFromName(dataSet):
  importName = "model_params.nyc_taxi_model_params_fast"
  print "Importing model params from %s" % importName
  try:
    importedModelParams = importlib.import_module(importName).MODEL_PARAMS
  except ImportError:
    raise Exception("No model params exist for '%s'. Run swarm first!"
                    % dataSet)
  return importedModelParams


def _getArgs():
  parser = OptionParser(usage="%prog PARAMS_DIR OUTPUT_DIR [options]"
                              "\n\nCompare TM performance with trivial predictor using "
                              "model outputs in prediction directory "
                              "and outputting results to result directory.")
  parser.add_option("-d",
                    "--dataSet",
                    type=str,
                    default='nyc_taxi',
                    dest="dataSet",
                    help="DataSet Name, choose from rec-center-hourly, nyc_taxi")

  parser.add_option("-p",
                    "--plot",
                    default=False,
                    dest="plot",
                    help="Set to True to plot result")

  parser.add_option("--stepsAhead",
                    help="How many steps ahead to predict. [default: %default]",
                    default=5,
                    type=int)

  (options, remainder) = parser.parse_args()
  print options

  return options, remainder


def getInputRecord(df, predictedField, i):
  inputRecord = {
    predictedField: float(df[predictedField][i]),
    "timeofday": float(df["timeofday"][i]),
    "dayofweek": float(df["dayofweek"][i]),
  }
  return inputRecord


def runMultiplePass(df, model, nMultiplePass, nTrain):
  """
  run CLA model through data record 0:nTrain nMultiplePass passes
  """
  predictedField = model.getInferenceArgs()['predictedField']
  print "run TM through the train data multiple times"
  for nPass in xrange(nMultiplePass):
    for j in xrange(nTrain):
      inputRecord = getInputRecord(df, predictedField, j)
      result = model.run(inputRecord)
      if j % 100 == 0:
        print " pass %i, record %i" % (nPass, j)
    # reset temporal memory
    model._getTPRegion().getSelf()._tfdr.reset()

  return model



def runMultiplePassSPonly(df, model, nMultiplePass, nTrain):
  """
  run CLA model SP through data record 0:nTrain nMultiplePass passes
  """

  predictedField = model.getInferenceArgs()['predictedField']
  print "run TM through the train data multiple times"
  for nPass in xrange(nMultiplePass):
    for j in xrange(nTrain):
      inputRecord = getInputRecord(df, predictedField, j)
      model._sensorCompute(inputRecord)
      model._spCompute()
      if j % 400 == 0:
        print " pass %i, record %i" % (nPass, j)

  return model


if __name__ == "__main__":
  print DESCRIPTION

  (_options, _args) = _getArgs()
  dataSet = _options.dataSet
  plot = _options.plot

  if dataSet == "rec-center-hourly":
    DATE_FORMAT = "%m/%d/%y %H:%M" # '7/2/10 0:00'
    predictedField = "kw_energy_consumption"
  elif dataSet == "nyc_taxi" or dataSet == "nyc_taxi_perturb" or dataSet =="nyc_taxi_perturb_baseline":
    DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
    predictedField = "passenger_count"
  else:
    raise RuntimeError("un recognized dataset")

  if dataSet == "nyc_taxi" or dataSet == "nyc_taxi_perturb" or dataSet =="nyc_taxi_perturb_baseline":
    modelParams = getModelParamsFromName("nyc_taxi")
  else:
    modelParams = getModelParamsFromName(dataSet)
  modelParams['modelParams']['clParams']['steps'] = str(_options.stepsAhead)

  print "Creating model from %s..." % dataSet

  # use customized CLA model
  model = CLAModel_custom(**modelParams['modelParams'])
  model.enableInference({"predictedField": predictedField})
  model.enableLearning()
  model._spLearningEnabled = True
  model._tpLearningEnabled = True


  inputData = "%s/%s.csv" % (DATA_DIR, dataSet.replace(" ", "_"))

  sensor = model._getSensorRegion()
  encoderList = sensor.getSelf().encoder.getEncoderList()
  if sensor.getSelf().disabledEncoder is not None:
    classifier_encoder = sensor.getSelf().disabledEncoder.getEncoderList()
    classifier_encoder = classifier_encoder[0]
  else:
    classifier_encoder = None

  _METRIC_SPECS = getMetricSpecs(predictedField, stepsAhead=_options.stepsAhead)
  metric = metrics.getModule(_METRIC_SPECS[0])
  metricsManager = MetricsManager(_METRIC_SPECS, model.getFieldInfo(),
                                  model.getInferenceType())

  if plot:
    plotCount = 1
    plotHeight = max(plotCount * 3, 6)
    fig = plt.figure(figsize=(14, plotHeight))
    gs = gridspec.GridSpec(plotCount, 1)
    plt.title(predictedField)
    plt.ylabel('Data')
    plt.xlabel('Timed')
    plt.tight_layout()
    plt.ion()

  print "Load dataset: ", dataSet
  df = pd.read_csv(inputData, header=0, skiprows=[1, 2])

  nMultiplePass = 5
  nTrain = 5000
  print " run SP through the first %i samples %i passes " %(nMultiplePass, nTrain)
  model = runMultiplePassSPonly(df, model, nMultiplePass, nTrain)
  model._spLearningEnabled = False


  maxBucket = classifier_encoder.n - classifier_encoder.w + 1
  likelihoodsVecAll = np.zeros((maxBucket, len(df)))

  prediction_nstep = None
  time_step = []
  actual_data = []
  patternNZ_track = []
  predict_data = np.zeros((_options.stepsAhead, 0))
  predict_data_ML = []
  negLL_track = []

  activeCellNum = []
  predCellNum = []
  predictedActiveColumnsNum = []
  trueBucketIndex = []
  sp = model._getSPRegion().getSelf()._sfdr
  spActiveCellsCount = np.zeros(sp.getColumnDimensions())

  # output = nupic_output.NuPICFileOutput([dataSet])

  startTime = time.time()
  timeHist = []
  for i in xrange(len(df)):
    startTimeI = time.time()
    inputRecord = getInputRecord(df, predictedField, i)
    result = model.run(inputRecord)
    endTimeI = time.time()
    timeHist.append(endTimeI - startTimeI)
  endTime = time.time()
  print "Time duration {}".format(endTime - startTime)




