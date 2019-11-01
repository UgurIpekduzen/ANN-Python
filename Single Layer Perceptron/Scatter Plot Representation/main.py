import random
import matplotlib.pyplot as plt
from perceptron import Perceptron
from inputs import Inputs
import numpy as np

def f(x):
    return 2 * x + 1

def createDataSet(intWidth, intHeight, intPoints):
    listDataSet = []
    for i in range(intPoints):
        x = random.uniform(-intWidth, intWidth)
        y = random.uniform(-intHeight, intHeight)
        # 2 giriş işleme girdiği zaman beklenen cevabın rastgele üretilmesi
        answer = 0 if y < f(x) else 1

        listDataSet.append(Inputs(x, y, answer))

    return listDataSet

if __name__ == '__main__':
    width = 500
    height = 500
    trainingPoints = 800
    testingPoints = 200

    listTrainingSet = createDataSet(width, height, trainingPoints)
    listTestingSet = createDataSet(width, height, testingPoints)
    print(np.size(listTrainingSet))

    objPerceptron = Perceptron(2, 0.01, 0.0, 0.0)

    score = []

    # Nesneyi eğit
    for t in listTrainingSet:
        objPerceptron.train(t.inputs, t.answer)

    listPosDataX = []
    listPosDataY = []
    listNegDataX = []
    listNegDataY = []
    # test girişlerinin sınıflandırılması
    for t in listTestingSet:
        guess = objPerceptron.predict(t.inputs)
        if guess > 0:
            listPosDataX.append(t.inputs[0])
            listPosDataY.append(t.inputs[1])
        else:
            listNegDataX.append(t.inputs[0])
            listNegDataY.append(t.inputs[1])

        # üretilen toplam test girişleri için yapılan doğru tahminlerin sayısının hesaplanması
        correct = -1 if t.inputs[1] < f(t.inputs[0]) else 1
        score.append(1 if guess is correct else 0)
    # Yapılan tahminlerin kesinliğinin hesaplanması
    print("Score:", sum(score), "/", len(score), "(" + str((float(sum(score)) / float(len(score))) * 100) + "% accuracy)")

    intMinX = - width
    intMaxX = width
    intMinY = f(intMinX)
    intMaxY = f(intMaxX)

    plt.plot([intMinX, intMaxX], [intMinY, intMaxY], linestyle='-', linewidth=3, color='green')

    #Grafikte sadece girilen genişlik içindeki veriler görüntülensin
    axes = plt.gca() #get current axes (gca)
    axes.set_xlim([intMinX, intMaxX])
    axes.set_ylim([intMinY / 2, intMaxY / 2])

    #Serpilme grafiğini çiz
    plt.scatter(listPosDataX, listPosDataY, marker='+', color='red')
    plt.scatter(listNegDataX, listNegDataY, marker='*', color='blue')

    plt.show()
