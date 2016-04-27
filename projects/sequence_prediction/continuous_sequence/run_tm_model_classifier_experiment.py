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
from optparse import OptionParser
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from ffnet import ffnet, mlgraph

from nupic.frameworks.opf.metrics import MetricSpec
from nupic.frameworks.opf.modelfactory import ModelFactory
from nupic.frameworks.opf.predictionmetricsmanager import MetricsManager
from nupic.frameworks.opf import metrics
from htmresearch.frameworks.opf.clamodel_custom import CLAModel_custom
from htmresearch.algorithms.cla_classifier_new import NewCLAClassifier
import nupic_output
from errorMetrics import *

matplotlib.rcParams.update({'figure.autolayout': True})
matplotlib.rcParams.update({'figure.autolayout': True})
matplotlib.use('TkAgg')

plt.ion()
DESCRIPTION = (
  "Run a HTM network with SP -> TM -> Classifier on NYC taxi data\n"
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



def printTPRegionParams(tpregion):
  """
  Note: assumes we are using TemporalMemory/TPShim in the TPRegion
  """
  tm = tpregion.getSelf()._tfdr
  print "------------PY  TemporalMemory Parameters ------------------"
  print "numberOfCols             =", tm.columnDimensions
  print "cellsPerColumn           =", tm.cellsPerColumn
  print "minThreshold             =", tm.minThreshold
  print "activationThreshold      =", tm.activationThreshold
  print "newSynapseCount          =", tm.maxNewSynapseCount
  print "initialPerm              =", tm.initialPermanence
  print "connectedPerm            =", tm.connectedPermanence
  print "permanenceInc            =", tm.permanenceIncrement
  print "permanenceDec            =", tm.permanenceDecrement
  print "predictedSegmentDecrement=", tm.predictedSegmentDecrement
  print



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
      if j % 500 == 0:
        print " pass %i, record %i" % (nPass, j)

  return model



def plotSPoutput(model, i):
  sp = model._getSPRegion().getSelf()._sfdr
  spOutput = model._getSPRegion().getOutputData('bottomUpOut')
  spActiveCellsCount = np.zeros(sp.getColumnDimensions())
  spActiveCellsCount[spOutput.nonzero()[0]] += 1

  activeDutyCycle = np.zeros(sp.getColumnDimensions(), dtype=np.float32)
  sp.getActiveDutyCycles(activeDutyCycle)
  overlapDutyCycle = np.zeros(sp.getColumnDimensions(), dtype=np.float32)
  sp.getOverlapDutyCycles(overlapDutyCycle)

  if i % 100 == 0 and i > 0:
    plt.figure(1)
    plt.clf()
    plt.subplot(2, 2, 1)
    plt.hist(overlapDutyCycle)
    plt.xlabel('overlapDutyCycle')

    plt.subplot(2, 2, 2)
    plt.hist(activeDutyCycle)
    plt.xlabel('activeDutyCycle-1000')

    plt.subplot(2, 2, 3)
    plt.hist(spActiveCellsCount)
    plt.xlabel('activeDutyCycle-Total')
    plt.draw()



def combineLikelihoodSingle(likelihoodsVecAll, targetStep, t):
  numBucket = likelihoodsVecAll[targetStep].shape[0]
  likelihood = np.ones(shape=(numBucket))
  for step in likelihoodsVecAll.keys():
    if t - (step - targetStep) < 0:
      continue
    likelihood *= likelihoodsVecAll[step][:, t - (step - targetStep)]

  if sum(likelihood) == 0:
    likelihood = likelihoodsVecAll[targetStep][:, t]

  likelihood = likelihood / sum(likelihood)
  return likelihood



def combineLikelihoods(likelihoodsVecAll, targetStep):
  likelihood = np.ones(shape=likelihoodsVecAll[targetStep].shape)
  for step in likelihoodsVecAll.keys():
    likelihood *= np.roll(likelihoodsVecAll[step], (step - targetStep), axis=1)
  # normalize likelihood
  likelihoodSum = np.sum(likelihood, 0)
  for i in xrange(len(likelihoodSum)):
    if likelihoodSum[i] == 0:
      likelihood[:, i] = likelihoodsVecAll[targetStep][:, i]
    likelihood[:, i] = likelihood[:, i] / likelihoodSum[i]

  return likelihood



def loadDataSet(dataSet):
  if dataSet == "rec-center-hourly":
    DATE_FORMAT = "%m/%d/%y %H:%M"  # '7/2/10 0:00'
    predictedField = "kw_energy_consumption"
  elif (dataSet == "nyc_taxi" or
            dataSet == "nyc_taxi_perturb" or
            dataSet == "nyc_taxi_perturb_baseline"):
    DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
    predictedField = "passenger_count"
  else:
    raise RuntimeError("unrecognized dataset")

  inputData = "%s/%s.csv" % (DATA_DIR, dataSet.replace(" ", "_"))
  print "Load dataset: ", dataSet
  df = pd.read_csv(inputData, header=0, skiprows=[1, 2])

  if (dataSet == "nyc_taxi" or
          dataSet == "nyc_taxi_perturb" or
          dataSet == "nyc_taxi_perturb_baseline"):
    modelParams = getModelParamsFromName("nyc_taxi")
  else:
    modelParams = getModelParamsFromName(dataSet)

  return (df, modelParams, predictedField)



if __name__ == "__main__":
  print DESCRIPTION

  (_options, _args) = _getArgs()
  dataSet = _options.dataSet
  plot = _options.plot

  (df, modelParams, predictedField) = loadDataSet(dataSet)

  # modelParams['modelParams']['clParams']['steps'] = str(_options.stepsAhead)
  targetStep = _options.stepsAhead
  modelParams['modelParams']['clParams']['steps'] = '5, 6, 7, 8, 9'
  modelParams['modelParams']['clParams']['implementation'] = 'py'
  print "Creating model from %s..." % dataSet

  # use customized CLA model
  model = CLAModel_custom(**modelParams['modelParams'])
  # clParams = {'alpha': 0.001,
  #             'numHistorySteps': 5,
  #             'steps': (5,),
  #             'numInputs': 65536,
  #             'verbosity': 0}
  #
  # model._getClassifierRegion().getSelf()._claClassifier = NewCLAClassifier(
  #   **clParams
  # )
  model.enableInference({"predictedField": predictedField})
  model.enableLearning()
  model._spLearningEnabled = True
  model._tpLearningEnabled = True

  printTPRegionParams(model._getTPRegion())

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
    # plt.tight_layout()
    plt.ion()

  nMultiplePass = 5
  nTrain = 5000
  print " run SP through the first %i samples %i passes " % (
  nMultiplePass, nTrain)
  model = runMultiplePassSPonly(df, model, nMultiplePass, nTrain)
  model._spLearningEnabled = False

  predictionSteps = model._getClassifierRegion().getSelf()._claClassifier.steps
  maxBucket = classifier_encoder.n - classifier_encoder.w + 1
  likelihoodsVecAll = {}
  for step in predictionSteps:
    likelihoodsVecAll[step] = np.zeros((maxBucket, len(df)))
  combinedLikelihood = np.zeros((maxBucket, len(df)))

  time_step = []
  actual_data = []
  patternNZ_track = []
  predict_data = np.zeros((_options.stepsAhead, 0))
  predict_data_ML = []

  activeCellNum = []
  predCellNum = []
  predictedActiveColumnsNum = []
  trueBucketIndex = []

  output = nupic_output.NuPICFileOutput([dataSet])

  tm = model._getTPRegion().getSelf()._tfdr
  activeTMCellBuffer = np.empty(shape=(tm.numberOfCells(), 0))

  # Create linear regression object
  conec = mlgraph((tm.numberOfCells(), 1))
  classificationNet = ffnet(conec)
  networkOutput = []

  for i in xrange(len(df)):
    inputRecord = getInputRecord(df, predictedField, i)
    tm = model._getTPRegion().getSelf()._tfdr

    prePredictiveCells = tm.predictiveCells
    prePredictiveColumn = np.array(list(prePredictiveCells)) / tm.cellsPerColumn

    result = model.run(inputRecord)
    trueBucketIndex.append(model._getClassifierInputRecord(inputRecord).bucketIndex)

    # analyze output of spatial pooler
    plotSPoutput(model, i)

    # analyze output of temporal memory
    tpOutput = tm.infActiveState['t']

    predictiveCells = tm.predictiveCells
    predCellNum.append(len(predictiveCells))
    predColumn = np.array(list(predictiveCells)) / tm.cellsPerColumn

    patternNZ = tpOutput.reshape(-1).nonzero()[0]
    activeColumn = patternNZ / tm.cellsPerColumn
    activeCellNum.append(len(patternNZ))

    predictedActiveColumns = np.intersect1d(prePredictiveColumn, activeColumn)
    predictedActiveColumnsNum.append(len(predictedActiveColumns))

    currentActiveCells = np.zeros(shape=(tm.numberOfCells(), 1))
    currentActiveCells[patternNZ] = 1

    activeTMCellBuffer = np.concatenate((activeTMCellBuffer, currentActiveCells), axis=1)
    time_step.append(i)
    actual_data.append(inputRecord[predictedField])

    result.metrics = metricsManager.update(result)

    if i % 1000 == 0 and i > 0:
      targetPrediction = np.roll(actual_data, -5)
      classificationNet.train_tnc(np.transpose(activeTMCellBuffer),
                                  targetPrediction,
                                  maxfun=500, messages=1)
    networkOutput.append(classificationNet(np.transpose(currentActiveCells)))


    if i % 100 == 0 and i > 0:
      negLL = result.metrics["multiStepBestPredictions:multiStep:"
                             "errorMetric='negativeLogLikelihood':steps=%d:window=1000:"
                             "field=%s" % (_options.stepsAhead, predictedField)]
      nrmse = result.metrics["multiStepBestPredictions:multiStep:"
                             "errorMetric='nrmse':steps=%d:window=1000:"
                             "field=%s" % (_options.stepsAhead, predictedField)]

      numActiveCell = np.mean(activeCellNum[-100:])
      numPredictiveCells = np.mean(predCellNum[-100:])
      numCorrectPredicted = np.mean(predictedActiveColumnsNum[-100:])

      print "After %i records, %d-step negLL=%f nrmse=%f ActiveCell %f " \
            "PredCol %f CorrectPredCol %f" % \
            (i, _options.stepsAhead, negLL, nrmse, numActiveCell,
             numPredictiveCells, numCorrectPredicted)

    for step in predictionSteps:
      bucketLL = result.inferences['multiStepBucketLikelihoods'][step]
      likelihoodsVec = np.zeros((maxBucket,))
      if bucketLL is not None:
        for (k, v) in bucketLL.items():
          likelihoodsVec[k] = v
      likelihoodsVecAll[step][0:len(likelihoodsVec), i] = likelihoodsVec

    likelihood = combineLikelihoodSingle(likelihoodsVecAll, targetStep, i)
    combinedLikelihood[:, i] = likelihood

    bucketValues = model._getClassifierRegion().getSelf()._claClassifier._actualValues
    mlPrediction = bucketValues[np.argmax(likelihood)]

    predict_data_ML.append(mlPrediction)
    # predict_data_ML.append(
    #   result.inferences['multiStepBestPredictions'][_options.stepsAhead])

    output.write([i], [inputRecord[predictedField]], [float(mlPrediction)])

    if plot and i > 500:
      # prepare data for display
      likelihood = combineLikelihoods(likelihoodsVecAll, _options.stepsAhead)
      # likelihood = likelihoodsVecAll[7]
      if i > 100:
        time_step_display = time_step[-500:-_options.stepsAhead]
        actual_data_display = actual_data[-500 + _options.stepsAhead:]
        predict_data_ML_display = predict_data_ML[-500:-_options.stepsAhead]
        likelihood_display = likelihood[:, i - 499:i - _options.stepsAhead + 1]
        xl = [(i) - 500, (i)]
      else:
        time_step_display = time_step
        actual_data_display = actual_data
        predict_data_ML_display = predict_data_ML
        likelihood_display = likelihood[:, :i + 1]
        xl = [0, (i)]

      plt.figure(2)
      plt.clf()
      plt.imshow(likelihood_display,
                 extent=(time_step_display[0], time_step_display[-1], 0, 40000),
                 interpolation='nearest', aspect='auto',
                 origin='lower', cmap='Reds')
      plt.plot(time_step_display, actual_data_display, 'k', label='Data')
      plt.plot(time_step_display, predict_data_ML_display, 'b',
               label='Best Prediction')
      plt.xlim(xl)
      plt.xlabel('Time')
      plt.ylabel('Prediction')
      plt.title('TM, useTimeOfDay=' + str(
        True) + ' ' + dataSet + ' test neg LL = ' + str(negLL))
      plt.draw()
  output.close()

  truth = np.roll(actual_data, -5)
  networkOutput = np.reshape(np.array(networkOutput), newshape=(len(networkOutput), ))
  plt.figure()
  plt.plot(networkOutput)
  plt.plot(predict_data_ML)
  plt.plot(truth)
  plt.xlim([2000, 2200])
  plt.legend(['Prediction-NN', 'Prediction-CLA', 'Truth'])

  plt.figure()
  plt.plot(predictionTrain)
  plt.plot(truth)
  plt.xlim([1000, 1200])
  plt.legend(['Prediction-NN', 'Truth'])

  # # calculate NRMSE
  # predData_TM_n_step = np.roll(np.array(predict_data_ML), _options.stepsAhead)
  # nTest = len(actual_data) - nTrain - _options.stepsAhead
  # NRMSE_TM = NRMSE(actual_data[nTrain:nTrain + nTest],
  #                  predData_TM_n_step[nTrain:nTrain + nTest])
  # print "NRMSE on test data: ", NRMSE_TM
  #
  # # calculate neg-likelihood
  # predictions = np.transpose(combinedLikelihood)
  # truth = np.roll(actual_data, -5)
  #
  # from nupic.encoders.scalar import ScalarEncoder as NupicScalarEncoder
  #
  # encoder = NupicScalarEncoder(w=1, minval=0, maxval=40000, n=22, forced=True)
  # from plot import computeLikelihood, plotAccuracy
  #
  # bucketIndex2 = []
  # negLL = []
  # minProb = 0.01
  # for i in xrange(len(truth)):
  #   bucketIndex2.append(np.where(encoder.encode(truth[i]))[0])
  #   outOfBucketProb = 1 - sum(predictions[i, :])
  #   prob = predictions[i, bucketIndex2[i]]
  #   if prob == 0:
  #     prob = outOfBucketProb
  #   if prob < minProb:
  #     prob = minProb
  #   negLL.append(-np.log(prob))
  #
  # negLL = computeLikelihood(predictions, truth, encoder)
  # negLL[:5000] = np.nan
  # x = range(len(negLL))
  # plt.figure()
  # plotAccuracy((negLL, x), truth, window=480, errorType='negLL')
  #
  # np.save('./result/' + dataSet + 'TMprediction.npy', predictions)
  # np.save('./result/' + dataSet + 'TMtruth.npy', truth)
