from sklearn import svm
import seaborn as sns
import matplotlib.pyplot as plt
from joblib import dump, load
import tensorflow as tf
import keras
import math
import numpy as np
import pandas as pd
import makedata as mdata
import bracket
import bracket_year

print(tf.version.VERSION)


def makeModel(input_len, output_len):
    # model = keras.Sequential([
    #    keras.layers.Dense(output_len, activation='sigmoid',
    #        input_shape=(input_len, )),
    #    # keras.layers.Dense(64, activation='sigmoid'),
    #    # keras.layers.Dense(64, activation='sigmoid'),
    #    # keras.layers.Dense(64, activation='sigmoid'),
    #    # keras.layers.Dense(output_len, activation='sigmoid'),
    # ])
    #
    # metrics = [
    #    keras.metrics.TruePositives(name='tp'),
    #    keras.metrics.FalsePositives(name='fp'),
    #    keras.metrics.TrueNegatives(name='tn'),
    #    keras.metrics.FalseNegatives(name='fn'),
    #    keras.metrics.BinaryAccuracy(name='accuracy'),
    #    keras.metrics.Precision(name='precision'),
    #    keras.metrics.Recall(name='recall'),
    #    keras.metrics.AUC(name='auc'),
    # ]
    #
    # model.compile(
    #    optimizer=keras.optimizers.Adam(lr=1e-3),
    #    loss=keras.losses.BinaryCrossentropy(),
    #    metrics=metrics)
    #
    # model.load_weights('saved_model/model2.ckpt')

    C = 200
    # C = 2**15
    # C = 0.1 ** 8
    # model = svm.SVC(kernel='poly', degree=3, gamma='auto', C=C, probability=True) #, verbose=True) #, max_iter=10_000)
    model = svm.SVC(kernel='rbf', probability=True)  # , verbose=True) #, max_iter=10_000)

    # model = load('model-svm1.joblib')

    return model


def train(model, m, training_inputs, training_outputs, testing_inputs, testing_outputs):
    print('SHAPES')
    print(training_inputs.shape, training_outputs.shape,
          testing_inputs.shape, testing_outputs.shape)

    model = model.fit(training_inputs, training_outputs.flatten())
    print('MODEL IS FIT')
    # model.fit(training_inputs, training_outputs, batch_size=128,
    #        epochs=128, verbose=1)

    # results = model.evaluate(testing_inputs, testing_outputs,
    #        batch_size=len(testing_inputs), verbose=0)
    # for name, value in zip(model.metrics_names, results):
    #    print(name, ': ', value)

    wTRank = []
    lTRank = []
    wLRank = []
    lLRank = []
    wGTRank = []
    lGTRank = []
    wGLRank = []
    lGLRank = []
    wTWinTen = []
    lTWinTen = []
    wSWinTen = []
    lSWinTen = []
    wTWinZero = []
    lTWinZero = []
    wSWinZero = []
    lSWinZero = []
    didWin = []

    for i in range(0):  # len(training_inputs)):
        index = 0
        # wTRank.append(training_inputs[i][index])
        # index += 1
        # lTRank.append(training_inputs[i][index])
        # index += 1
        # wLRank.append(training_inputs[i][index])
        # index += 1
        # lLRank.append(training_inputs[i][index])
        # index += 1
        # wGTRank.append(training_inputs[i][index])
        # index += 1
        # lGTRank.append(training_inputs[i][index])
        # index += 1
        # wGLRank.append(training_inputs[i][index])
        # index += 1
        # lGLRank.append(training_inputs[i][index])
        # index += 1
        wTWinTen.append(training_inputs[i][index])
        index += 1
        lTWinTen.append(training_inputs[i][index])
        index += 1
        wSWinTen.append(training_inputs[i][index])
        index += 1
        lSWinTen.append(training_inputs[i][index])
        index += 1
        wTWinZero.append(training_inputs[i][index])
        index += 1
        lTWinZero.append(training_inputs[i][index])
        index += 1
        wSWinZero.append(training_inputs[i][index])
        index += 1
        lSWinZero.append(training_inputs[i][index])
        index += 1
        didWin.append(training_outputs[i][0])
        index += 1

    # df = pd.DataFrame({
    #    'WinTeamRank': wTRank,
    #    'LossTeamRank': lTRank,
    #    'WinLeagueRank': wLRank,
    #    'LossLeagueRank': lLRank,
    #    'WinTeamGoalsRank': wGTRank,
    #    'LossTeamGoalsRank': lGTRank,
    #    'WinLeagueGoalsRank': wGLRank,
    #    'LossLeagueGoalsRank': lGLRank,
    #    #'WinTSeasonWinPercentage10': wTWinTen,
    #    #'LossTWinPercentage10': lTWinTen,
    #    #'WinSWinPercentage10': wSWinTen,
    #    #'LossSWinPercentage10': lSWinTen,
    #    #'WinTWinPercentage': wTWinZero,
    #    #'LossTWinPercentage': lTWinZero,
    #    #'WinSWinPercentage': wSWinZero,
    #    #'LossSWinPercentage': lSWinZero,
    #    'didWin': didWin,
    #    })

    # plot = sns.pairplot(df, hue='didWin')
    # plot.show()
    # plot.savefig('plot2.png')

    # model.save_weights("saved_model/model2.ckpt")
    # dump(model, 'model-svm2.joblib')

    return m, model


def predictGame(m, model, year):
    def predict(team1, team2):
        inputs = mdata.getInputs(m, year, team1, team2)

        won = model.predict_proba(inputs)[0][1]
        los = model.predict_proba(inputs)[0][0]
        prob = won / (won + los)
        return prob
        # return (math.tanh(30*(prob-0.5))+1)/2.0
        # return model.predict(inputs)[0][0]

    return predict


def convertTeamToStr(m):
    def convert(teamId):
        if teamId in m.teamid_to_teamname:
            return m.teamid_to_teamname[teamId]
        return teamId + '-unknown'

    return convert


def testStuff(training_inputs, training_outputs, transform):
    correct = 0
    wrong = 0

    for i in range(len(training_inputs)):
        h = transform(training_inputs[i][0], training_inputs[i][2])
        a = transform(training_inputs[i][1], training_inputs[i][3])

        if h > a and training_outputs[i][0] == 1 or h < a and training_outputs[i][0] == 0:
            correct += 1
        else:
            wrong += 1

    return correct, wrong


if __name__ == "__main__":
    m, training_inputs, training_outputs, test_inputs, test_outputs = mdata.getData()
    model = makeModel(training_inputs.shape[1], training_outputs.shape[1])

    _, model = train(model, m, training_inputs, training_outputs, test_inputs, test_outputs)
    # model = makeModel(inputs.shape[1], outputs.shape[1])

    # May change bracketYear to reflect the final bracket you want
    #       and change the year to the year you want.
    b = bracket.Bracket(bracket_year.the2016Bracket, predictGame(m, model, 2016), convertTeamToStr(m))
    b.playTourne()
    b = bracket.Bracket(bracket_year.the2018Bracket, predictGame(m, model, 2018), convertTeamToStr(m))
    b.playTourne()
    b = bracket.Bracket(bracket_year.the2019Bracket, predictGame(m, model, 2019), convertTeamToStr(m))
    b.playTourne()
    # b = bracket.Bracket(bracketYear.the2021Bracket,
    #        predictGame(m, model, 2020), convertTeamToStr(m))
    # b.playTourne()

    b = bracket.Bracket(bracket_year.the2021Bracket, predictGame(m, model, 2021), convertTeamToStr(m))
    b.playTourne()

    b = bracket.Bracket(bracket_year.the2021SecondChanceBracket, predictGame(m, model, 2021), convertTeamToStr(m))
    b.playTourne()

    b = bracket.Bracket(bracket_year.the2022Bracket, predictGame(m, model, 2022), convertTeamToStr(m))
    b.playTourne()

