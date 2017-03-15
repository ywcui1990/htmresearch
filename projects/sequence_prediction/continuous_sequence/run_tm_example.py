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

from nupic.frameworks.opf.metrics import MetricSpec
from nupic.frameworks.opf.modelfactory import ModelFactory
from nupic.frameworks.opf.predictionmetricsmanager import MetricsManager
from nupic.frameworks.opf import metrics
from htmresearch.frameworks.opf.clamodel_custom import CLAModel_custom
import nupic_output

import pandas as pd
from htmresearch.support.sequence_learning_utils import *

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
  importName = "model_params.%s_model_params" % (
    dataSet.replace(" ", "_").replace("-", "_")
  )
  print "Importing model params from %s" % importName
  try:
    importedModelParams = importlib.import_module(importName).MODEL_PARAMS
  except ImportError:
    raise Exception("No model params exist for '%s'. Run swarm first!"
                    % dataSet)
  return importedModelParams



def getInputRecord(df, predictedField, i):
  inputRecord = {
    predictedField: float(df[predictedField][i]),
    "timeofday": float(df["timeofday"][i]),
    "dayofweek": float(df["dayofweek"][i]),
  }
  return inputRecord


def printTPRegionParams(tpregion):
  """
  Note: assumes we are using TemporalMemory/TPShim in the TPRegion
  """
  tm = tpregion.getSelf()._tfdr
  print "------------PY  TemporalMemory Parameters ------------------"
  print "numberOfCols             =", tm.getColumnDimensions()
  print "cellsPerColumn           =", tm.getCellsPerColumn()
  print "minThreshold             =", tm.getMinThreshold()
  print "activationThreshold      =", tm.getActivationThreshold()
  print "newSynapseCount          =", tm.getMaxNewSynapseCount()
  print "initialPerm              =", tm.getInitialPermanence()
  print "connectedPerm            =", tm.getConnectedPermanence()
  print "permanenceInc            =", tm.getPermanenceIncrement()
  print "permanenceDec            =", tm.getPermanenceDecrement()
  print "predictedSegmentDecrement=", tm.getPredictedSegmentDecrement()
  print



if __name__ == "__main__":

  dataSet = "nyc_taxi"

  DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
  predictedField = "passenger_count"
  modelParams = getModelParamsFromName("nyc_taxi")
  stepsAhead = 5

  print "Creating model from %s..." % dataSet

  # use customized CLA model
  model = CLAModel_custom(**modelParams['modelParams'])
  model.enableInference({"predictedField": predictedField})
  model.enableLearning()
  model._spLearningEnabled = False # disable SP learning
  model._tpLearningEnabled = True

  printTPRegionParams(model._getTPRegion())

  inputData = "%s/%s.csv" % (DATA_DIR, dataSet.replace(" ", "_"))

  _METRIC_SPECS = getMetricSpecs(predictedField, stepsAhead=stepsAhead)
  metric = metrics.getModule(_METRIC_SPECS[0])
  metricsManager = MetricsManager(_METRIC_SPECS, model.getFieldInfo(),
                                  model.getInferenceType())

  print "Load dataset: ", dataSet
  df = pd.read_csv(inputData, header=0, skiprows=[1, 2])

  prediction_nstep = None
  sp = model._getSPRegion().getSelf()._sfdr
  tp = model._getTPRegion()
  tm = tp.getSelf()._tfdr

  output = nupic_output.NuPICFileOutput([dataSet])

  activeCellNum = []
  predCellNum = []
  predictedActiveColumnsNum = []

  for i in xrange(2000):
    inputRecord = getInputRecord(df, predictedField, i)
    prePredictiveCells = tm.getPredictiveCells()
    prePredictiveColumn = np.array(list(prePredictiveCells)) / tm.cellsPerColumn

    result = model.run(inputRecord)

    tp = model._getTPRegion()
    tm = tp.getSelf()._tfdr
    tpOutput = tm.infActiveState['t']

    predictiveCells = tm.getPredictiveCells()
    predCellNum.append(len(predictiveCells))
    predColumn = np.array(list(predictiveCells))/ tm.cellsPerColumn

    patternNZ = tpOutput.reshape(-1).nonzero()[0]
    activeCellNum.append(len(patternNZ))
    activeColumn = patternNZ / tm.cellsPerColumn

    predictedActiveColumns = np.intersect1d(prePredictiveColumn, activeColumn)
    predictedActiveColumnsNum.append(len(predictedActiveColumns))

    result.metrics = metricsManager.update(result)

    negLL = result.metrics["multiStepBestPredictions:multiStep:"
               "errorMetric='negativeLogLikelihood':steps=%d:window=1000:"
               "field=%s"%(stepsAhead, predictedField)]
    if i % 100 == 0 and i>0:
      negLL = result.metrics["multiStepBestPredictions:multiStep:"
               "errorMetric='negativeLogLikelihood':steps=%d:window=1000:"
               "field=%s"%(stepsAhead, predictedField)]
      nrmse = result.metrics["multiStepBestPredictions:multiStep:"
               "errorMetric='nrmse':steps=%d:window=1000:"
               "field=%s"%(stepsAhead, predictedField)]

      numActiveCell = np.mean(activeCellNum[-100:])
      numPredictiveCells = np.mean(predCellNum[-100:])
      numCorrectPredicted = np.mean(predictedActiveColumnsNum[-100:])

      print "After %i records, %d-step negLL=%f nrmse=%f ActiveCell %f PredCol %f CorrectPredCol %f" % \
            (i, stepsAhead, negLL, nrmse, numActiveCell,
             numPredictiveCells, numCorrectPredicted)
