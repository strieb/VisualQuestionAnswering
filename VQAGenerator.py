import numpy as np
import pickle
import json
from keras.utils import Sequence
from collections import Counter
from PIL import Image, ImageDraw
import random
import skimage.io as io
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import os
from keras.preprocessing.image import load_img

class VQAGenerator(Sequence):
    def __init__(self, train=True, batchSize=32, dataDir='C:/ml/VQA', predict=False, imageType=None):
        self.dataSubType = 'train2014' if train else 'val2014'
        self.dataDir = dataDir
        self.batchSize = batchSize
        self.predict = predict
        self.imageType = imageType
        self.train = train
        databaseFile = '%s/Database/%s.pickle' % (dataDir, self.dataSubType)
        imageIndexFile = '%s/Database/%simageindex.json' % (dataDir, self.dataSubType)
        imagesFile = '%s/Database/%simages.npy' % (dataDir, self.dataSubType)
        questionsEncFile = '%s/Database/questions.json' % (dataDir)
        answersEncFile = '%s/Database/answers.json' % (dataDir)
        gloveFile = '%s/Database/glove100.pickle' % (dataDir)
        self.imagesDirectory = '%s/Images/%s/%s/' % (dataDir,self.dataSubType, self.imageType)
        complementaryFile = '%s/Database/v2_mscoco_train2014_complementary_pairs.json' % (dataDir)
        self.resultsFile =  '%s/Results/results.json' % (dataDir)
        
        with open(databaseFile, 'rb') as fp:
            self.database = pickle.load(fp)

        with open(gloveFile, 'rb') as fp:
            self.gloveIndex = pickle.load(fp)

        if self.imageType == None:
            with open(imageIndexFile, 'rb') as fp:
                self.imageindex = json.load(fp)
            self.images = np.load(imagesFile)

        with open(questionsEncFile, 'rb') as fp:
            self.questionEncoding = json.load(fp)
        with open(answersEncFile, 'rb') as fp:
            self.answerEncoding = json.load(fp)

        with open(complementaryFile, 'rb') as fp:
            self.complementaries = json.load(fp)

        self.answerLength = len(self.answerEncoding)
        self.questionLength = len(self.questionEncoding)
        self.on_epoch_end()
      
    def on_epoch_end(self):
        if self.train:
            # random.shuffle(self.complementaries)
            # complementariesFlat = [index for both in self.complementaries for index in both]
            # questionIDs = {self.database['ids'][i]: i for i in range(len(self.database['ids']))}
            # complementariesIds = [questionIDs[index] for index in complementariesFlat]
            allIds = [i for i in range(len(self.database['answers']))]
            # diff = list(set(allIds)-set(complementariesIds))
            # random.shuffle(diff)
            # self.good = diff + complementariesIds
            self.good = allIds
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
        if self.imageType == None:
            idx = self.imageindex[str(imageId)]
            return self.images[idx]
        else:
            return np.load(self.imagesDirectory+str(imageId)+'.npy')

    def gloveEncoding(self):
        mat = np.random.rand(self.questionLength + 2,100)
        inv_tokens = {v: k for k, v in self.questionEncoding.items()}
        for i in range(self.questionLength):
            token = inv_tokens[i]

            if token in self.gloveIndex:
                mat[i+2] = self.gloveIndex[token]
        return mat


    def __getitem__(self, idx):
        maxQuestionLength = 14
        offset = idx * self.batchSize
        idxs = self.good[offset: offset + self.batchSize]
        size = len(idxs)
        imageBatch = np.ndarray((size,24, 2048), dtype=np.float32)
        questionSpecialBatch = np.zeros((size, maxQuestionLength), dtype=np.int32)
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
                if t >= 14:
                    break
                token = str.lower(token)
                if token in self.questionEncoding:
                    questionSpecialBatch[i,t] = self.questionEncoding[token] + 2
                else:
                    questionSpecialBatch[i,t] = 1
                t = t + 1
            imageBatch[i, :] = self.getImage(i + offset)

        input = [questionSpecialBatch, imageBatch]
        if self.predict:
            return input
        else:
            return [input, answerBatch]

    def print(self, idx,pred,heat):
        print('Question: ' + str(' '.join(self.database['questions'][idx])))
        print('Answers: ' + str(self.database['answers'][idx]))
        top = [(i,pred[i]) for i in range(len(pred))]
        top = sorted(top, key=lambda entry: entry[1])
        inv_map = {v: k for k, v in self.answerEncoding.items()}
        for entry in top[-5:]:
            print('Result '+str(entry[0])+': '+str(entry[1])+", "+inv_map[entry[0]])


        imageId = self.database['image_ids'][idx]
        imgPath = self.dataDir+'/Images/'+self.dataSubType+'/COCO_' + self.dataSubType + '_' + str(imageId).zfill(12) + '.jpg'
        if os.path.isfile(imgPath):
            # img = Image.open(imgPath)
            img = load_img(imgPath)
            width, height = img.size
            if(width < height):
                img = img.resize((427, 619), resample=Image.BICUBIC)
                heat = heat.reshape((4, 6))
            else: 
                img = img.resize((619, 427), resample=Image.BICUBIC)
                heat = heat.reshape((6, 4))

            draw = ImageDraw.Draw(img,'RGBA')
            for x in range(heat.shape[0]):
                for y in range(heat.shape[1]):
                    r = min(heat[x,y] * 15,1)
                    gb = max(min(heat[x,y] * 15 - 1,1),0)
                    draw.ellipse(((x*96+22+40,y*96+22+40),(x*96+22+56,y*96+22+56)),fill=(int(r * 255), int(gb * 255), int(gb * 255), int(r * 255)))
            # plt.imshow(heat, cmap='hot',norm=colors.Normalize(), interpolation='nearest')
            plt.imshow(img)
            plt.axis('off')
            plt.show()

    def evaluate(self, predictions):
        inv_encodings = {v: k for k, v in self.answerEncoding.items()}
        # predictions[:,2047] = 0
        max = np.argmax(predictions, axis=1)
        length = len(max)
        score = 0
        results = []
        for i in range(length):
            answer = inv_encodings[max[i]]
            question_id = self.database['ids'][i]
            results.append({'question_id': question_id, 'answer': answer})
            corrects = np.sum([1 if gtAnswer == answer else 0 for gtAnswer in self.database['answers'][i]])
            score += min(corrects, 3.0) / 3.0
        print('Accuracy: ' + str(score / length))
        with open(self.resultsFile, 'w') as fp:
            json.dump(results, fp)
