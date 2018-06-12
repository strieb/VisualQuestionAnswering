import numpy as np
import pickle
import json
from keras.utils import Sequence
from collections import Counter
from PIL import Image


class VQAGenerator(Sequence):
    def __init__(self, train=True, batchSize=32, dataDir='C:/ml/VQA'):
        self.dataSubType = 'train2014' if train else 'val2014'
        self.dataDir = dataDir
        self.batchSize = batchSize
        databaseFile = '%s/Database/%s' % (dataDir, self.dataSubType)
        questionsEncFile = '%s/Database/questions.json' % (dataDir)
        answersEncFile = '%s/Database/answers.json' % (dataDir)

        with open(databaseFile, 'rb') as fp:
            self.database = pickle.load(fp)

        with open(databaseFile+"imageindex.json", 'rb') as fp:
            self.imageindex = json.load(fp)

        self.images = np.load(databaseFile+"images.npy")

        with open(questionsEncFile, 'rb') as fp:
            self.questionEncoding = json.load(fp)
        with open(answersEncFile, 'rb') as fp:
            self.answerEncoding = json.load(fp)

        self.answerLength = len(self.answerEncoding)
        self.questionLength = len(self.answerEncoding)

        if train:
            self.good = [i for i in range(len(self.database['answers'])) if self.getAnswer(i) in self.answerEncoding]
        else:
            self.good = [i for i in range(len(self.database['answers']))]

    def __len__(self):
        return int(np.ceil(len(self.good)/float(self.batchSize)))

    def getAnswer(self, i):
        answers = self.database['answers'][i]
        counter = Counter(answers).most_common(1)
        return counter[0][0]

    def getQuestion(self, i):
        question = self.database['questions'][i]
        return str.split(question)

    def getImage(self, i):
        imageId = self.database['image_ids'][i]
        idx = self.imageindex[str(imageId)]
        return self.images[idx]

    def __getitem__(self, idx):

        offset = idx * self.batchSize
        idxs = self.good[offset: offset + self.batchSize]
        size = len(idxs)
        imageBatch = np.zeros((size, 2048), dtype=np.float32)
        questionBatch = np.zeros((size, self.questionLength), dtype=np.int8)
        answerBatch = np.zeros((size, self.answerLength), dtype=np.int8)

        for i in range(size):
            answer = self.getAnswer(i + offset)
            if answer in self.answerEncoding:
                answerBatch[i, self.answerEncoding[answer]] = 1

            question = self.getQuestion(i + offset)
            for token in question:
                if token in self.questionEncoding:
                    questionBatch[i, self.questionEncoding[token]] = 1
            imageBatch[i, :] = self.getImage(i + offset)

        return [[questionBatch, imageBatch], answerBatch]

    def single(self, idx):
        offset = idx
        size = 1
        imageBatch = np.zeros((size, 2048), dtype=np.float32)
        questionBatch = np.zeros((size, self.questionLength), dtype=np.int8)
        answerBatch = np.zeros((size, self.answerLength), dtype=np.int8)

        for i in range(size):
            print('Question: ' + str(self.database['questions'][i+offset]))
            print('Answers: ' + str(self.database['answers'][i+offset]))
            imageId = self.database['image_ids'][i+offset]
            imgPath = self.dataDir+'/Images/'+self.dataSubType+'/COCO_' + self.dataSubType + '_' + str(imageId).zfill(12) + '.jpg'
            im = Image.open(imgPath)
            im.show()

            question = self.getQuestion(i + offset)
            for token in question:
                if token in self.questionEncoding:
                    questionBatch[i, self.questionEncoding[token]] = 1
            imageBatch[i, :] = self.getImage(i + offset)

        return [questionBatch, imageBatch]
