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

from Environment import DATADIR
from VQAConfig import VQAConfig

class VQAGenerator(Sequence):
    def __init__(self, train,  predict, config: VQAConfig):
        self.config = config
        self.augmentations = config.augmentations
        self.imageFeatureSize = config.imageFeaturemapSize
        self.imageFeatureStateSize = config.imageFeatureChannels
        self.dataSubType = 'train2014' if train else 'val2014'
        self.batchSize = config.batchSize
        self.predict = predict
        self.imageType = config.imageType
        self.train = train
        databaseFile = '%s/Database/%s.pickle' % (DATADIR, self.dataSubType)
        imageIndexFile = '%s/Database/%simageindex.json' % (DATADIR, self.dataSubType)
        imagesFile = '%s/Database/%simages.npy' % (DATADIR, self.dataSubType)
        questionsEncFile = '%s/Database/questions.json' % (DATADIR)
        answersEncFile = '%s/Database/answers.json' % (DATADIR)
        self.imagesDirectory = '%s/Images/%s/%s/' % (DATADIR,self.dataSubType, self.imageType)
        self.augmentationDirectory = '%s/Images/%s/%s/' % (DATADIR,self.dataSubType, 'augmented_res_24')
        complementaryFile = '%s/Database/v2_mscoco_train2014_complementary_pairs.json' % (DATADIR)
        self.resultsFile =  '%s/Results/results.json' % (DATADIR)

        with open(databaseFile, 'rb') as fp:
            self.database = pickle.load(fp)

        if self.imageType == None:
            with open(imageIndexFile, 'r') as fp:
                self.imageindex = json.load(fp)
            self.images = np.load(imagesFile)

        with open(questionsEncFile, 'r',) as fp:
            self.questionEncoding = json.load(fp)
        with open(answersEncFile, 'r') as fp:
            self.answerEncoding = json.load(fp)

        with open(complementaryFile, 'r') as fp:
            self.complementaries = json.load(fp)

        self.answerLength = len(self.answerEncoding)
        self.questionLength = len(self.questionEncoding)
        self.on_epoch_end()
      
    def on_epoch_end(self):
        # if self.balanced and self.train:
        #     random.shuffle(self.complementaries)
        #     complementariesFlat = [index for both in self.complementaries for index in both]
        #     questionIDs = {self.database['ids'][i]: i for i in range(len(self.database['ids']))}
        #     complementariesIds = [questionIDs[index] for index in complementariesFlat]
        #     allIds = [i for i in range(len(self.database['answers']))]
        #     diff = list(set(allIds)-set(complementariesIds))
        #     random.shuffle(diff)
        #     self.good = diff + complementariesIds
        if self.train:
            allIds = [i for i in range(len(self.database['answers']))]
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
            if self.augmentations == None or not self.train:
                return np.load(self.imagesDirectory+str(imageId)+'.npy')
            else:
                randNumber = random.randint(0,self.augmentations-1)
                return np.load(self.augmentationDirectory+str(imageId)+'_'+str(randNumber) +'.npy')

    
    def getImageFromDirectory(self, i, directory):
        idx = self.good[i]
        imageId = self.database['image_ids'][idx]
        return np.load(directory+str(imageId)+'.npy')

    def gloveEncoding(self):
        gloveFile = '%s/Database/%s.pickle' % (DATADIR, self.config.gloveName)
        with open(gloveFile, 'rb') as fp:
            gloveIndex = pickle.load(fp)

        inside = 0
        all = 0
        # 1 - start
        # 2 - end
        # 3 - unknown
        mat = np.random.rand(self.questionLength + 4, self.config.gloveSize)
        inv_tokens = {v: k for k, v in self.questionEncoding.items()}
        for i in range(self.questionLength):
            token = inv_tokens[i]
            all += 1
            if token in gloveIndex:
                inside += 1
                mat[i+4] = gloveIndex[token]
        print("tokens")
        print(inside)
        print(all)
        return mat


    def __getitem__(self, idx):
        maxQuestionLength = 14
        offset = idx * self.batchSize
        idxs = self.good[offset: offset + self.batchSize]
        size = len(idxs)
        imageBatch = np.ndarray((size, self.imageFeatureSize, self.imageFeatureStateSize), dtype=np.float32)
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
            t = 1
            for token in question:
                if t >= maxQuestionLength:
                    break
                token = str.lower(token)
                if token in self.questionEncoding:
                    questionSpecialBatch[i,t] = self.questionEncoding[token] + 4
                else:
                    questionSpecialBatch[i,t] = 3
                t = t + 1

            imageBatch[i,:, :] = self.getImage(i + offset)

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
        imgPath = DATADIR+'/Images/'+self.dataSubType+'/COCO_' + self.dataSubType + '_' + str(imageId).zfill(12) + '.jpg'
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
            cmap = plt.get_cmap('magma')

            draw = ImageDraw.Draw(img,'RGBA')
            for x in range(heat.shape[0]):
                for y in range(heat.shape[1]):
                    c = cmap(heat[x,y] * 12)
                    draw.ellipse(((x*96+22+40,y*96+22+40),(x*96+22+56,y*96+22+56)),fill=(int(c[0]*255),int(c[1]*255),int(c[2]*255),int(c[3]*255)))
                    # r = max(min(heat[x,y] +1,1),0)
                    # bg = max(min(heat[x,y],1),0)
                    # draw.ellipse(((x*96+22+40,y*96+22+40),(x*96+22+56,y*96+22+56)),fill=(int((r) * 255), int(bg * 255),int(0 * 255) , int(r * 255)))
            # plt.imshow(heat, cmap='hot',norm=colors.Normalize(), interpolation='nearest')
            plt.imshow(img)
            plt.axis('off')
            plt.show()

    def evaluate(self, predictions):
        inv_encodings = {v: k for k, v in self.answerEncoding.items()}
        # predictions[:,2047] = 0
        # max = np.argmax(predictions, axis=1)
        max = predictions
        length = len(max)
        score = 0
        results = []
        for i in range(length):
            answer = inv_encodings[max[i]]
            question_id = self.database['ids'][i]
            results.append({'question_id': question_id, 'answer': answer})
            corrects = np.sum([1 if gtAnswer == answer else 0 for gtAnswer in self.database['answers'][self.good[i]]])
            score += min(corrects, 3.0) / 3.0
        print('Accuracy: ' + str(score / length))
        with open(self.resultsFile, 'w') as fp:
            json.dump(results, fp)
        return score / length
