import numpy as np
import pickle
import json
from keras.utils import Sequence
from collections import Counter
from PIL import Image
import random
import skimage.io as io
import matplotlib.pyplot as plt
import os

class VQAGenerator(Sequence):
    def __init__(self, train=True, batchSize=32, dataDir='C:/ml/VQA', predict=False):
        self.dataSubType = 'train2014' if train else 'val2014'
        self.dataDir = dataDir
        self.batchSize = batchSize
        self.predict = predict
        databaseFile = '%s/Database/%s.pickle' % (dataDir, self.dataSubType)
        imageIndexFile = '%s/Database/%simageindex.json' % (dataDir, self.dataSubType)
        imagesFile = '%s/Database/%simages.npy' % (dataDir, self.dataSubType)
        questionsEncFile = '%s/Database/questions.json' % (dataDir)
        answersEncFile = '%s/Database/answers.json' % (dataDir)
        gloveFile = '%s/Database/glove100.pickle' % (dataDir)

        complementaryFile = '%s/Database/v2_mscoco_train2014_complementary_pairs.json' % (dataDir)

        with open(databaseFile, 'rb') as fp:
            self.database = pickle.load(fp)

        with open(gloveFile, 'rb') as fp:
            self.gloveIndex = pickle.load(fp)

        with open(imageIndexFile, 'rb') as fp:
            self.imageindex = json.load(fp)

        self.images = np.load(imagesFile)

        with open(questionsEncFile, 'rb') as fp:
            self.questionEncoding = json.load(fp)
        with open(answersEncFile, 'rb') as fp:
            self.answerEncoding = json.load(fp)

        self.answerLength = len(self.answerEncoding)
        self.questionLength = len(self.answerEncoding)

        if train:
            with open(complementaryFile, 'rb') as fp:
                complementaries = json.load(fp)
            random.shuffle(complementaries)
            questionIDs = {self.database['ids'][i]: i for i in range(len(self.database['ids']))}
            complementariesFlat = [index for both in complementaries for index in both]
            # self.good = [questionIDs[index] for index in complementariesFlat]
            # self.good = [i for i in range(len(self.database['answers'])) if self.getAnswer(i) in self.answerEncoding]
            self.good = [i for i in range(len(self.database['answers']))]
            random.shuffle(self.good)
        else:
            self.good = [i for i in range(len(self.database['answers']))]

    def __len__(self):
        return int(np.ceil(len(self.good)/float(self.batchSize)))

    def getAnswer(self, i):
        idx = self.good[i]
        answers = self.database['answers'][idx]
        counter = Counter(answers).most_common(1)
        return counter[0][0]

    def getAnswers(self, i):
        idx = self.good[i]
        answers = self.database['answers'][idx]
        ret = []
        for answer in answers:
            if answer in self.answerEncoding:
                ret = ret + [answer]
        # if len(ret) == 0:
        #     return ret
        # most = Counter(ret).most_common(1)
        # if(most[0][1] >= 5):
        #     return[most[0][0]] * 10
        return ret

    def getQuestion(self, i):
        idx = self.good[i]
        question = self.database['questions'][idx]
        return question

    def getImage(self, i):
        idx = self.good[i]
        imageId = self.database['image_ids'][idx]
        idx = self.imageindex[str(imageId)]
        return self.images[idx]

    def __getitem__(self, idx):
        questionLength = 14

        offset = idx * self.batchSize
        idxs = self.good[offset: offset + self.batchSize]
        size = len(idxs)
        imageBatch = np.zeros((size, 2048), dtype=np.float32)
        questionBatch = np.zeros((size, questionLength, 100), dtype=np.float32)
        # questionSpecialBatch = np.zeros((size, self.questionLength), dtype=np.float32)
        answerBatch = np.zeros((size, self.answerLength), dtype=np.float32)

        for i in range(size):
            answers = self.getAnswers(i + offset)
            for answer in answers:
                answerBatch[i, self.answerEncoding[answer]] += 0.1
                if answerBatch[i, self.answerEncoding[answer]] > 1:
                    answerBatch[i, self.answerEncoding[answer]] = 1
            # answer = self.getAnswer(i + offset)
            # if answer in self.answerEncoding:
            #     answerBatch[i, self.answerEncoding[answer]] = 1

            question = self.getQuestion(i + offset)
            t = 0
            for token in question:
                token = str.lower(token)
                # if token in self.questionEncoding:
                #     questionSpecialBatch[i,self.questionEncoding[token]] = 1

                if token in self.gloveIndex and t < questionLength:
                    questionBatch[i, t, 0:100] = self.gloveIndex[token]
                    t += 1
            imageBatch[i, :] = self.getImage(i + offset)
        input = [questionBatch, imageBatch]
        if self.predict:
            return input
        else:
            return [input, answerBatch]

    def print(self, idx,pred):
        print('Question: ' + str(self.database['questions'][idx]))
        print('Answers: ' + str(self.database['answers'][idx]))
        top = [(i,pred[i]) for i in range(len(pred))]
        top = sorted(top, key=lambda entry: entry[1])
        inv_map = {v: k for k, v in self.answerEncoding.items()}
        for entry in top[-5:]:
            print('Result '+str(entry[0])+': '+str(entry[1])+", "+inv_map[entry[0]])


        imageId = self.database['image_ids'][idx]
        imgPath = self.dataDir+'/Images/'+self.dataSubType+'/COCO_' + self.dataSubType + '_' + str(imageId).zfill(12) + '.jpg'
        if os.path.isfile(imgPath):
            I = io.imread(imgPath)
            plt.imshow(I)
            plt.axis('off')
            plt.show()



    def evaluate(self, predictions):
        inv_encodings = {v: k for k, v in self.answerEncoding.items()}
        # predictions[:,2047] = 0
        max = np.argmax(predictions, axis=1)
        length = len(max)
        score = 0
        for i in range(length):
            answer = inv_encodings[max[i]]
            corrects = np.sum([1 if gtAnswer == answer else 0 for gtAnswer in self.database['answers'][i]])
            score += min(corrects, 3.0) / 3.0
        print('Accuracy: ' + str(score / length))
