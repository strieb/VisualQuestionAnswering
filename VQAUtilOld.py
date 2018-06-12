from VQA.PythonHelperTools.vqaTools.vqa import VQA
from collections import Counter
import re
import numpy as np
import json
from keras.preprocessing.image import load_img, img_to_array


class VQAUtil:

    dataDir = 'C:/ml/VQA'
    versionType = 'v2_'  # this should be '' when using VQA v2.0 dataset
    taskType = 'OpenEnded'  # 'OpenEnded' only for v2.0. 'OpenEnded' or 'MultipleChoice' for v1.0
    dataType = 'mscoco'  # 'mscoco' only for v1.0. 'mscoco' for real and 'abstract_v002' for abstract for v1.0.
    dataSubType = 'train2014'
    annFile = '%s/Annotations/%s%s_%s_annotations.json' % (dataDir, versionType, dataType, dataSubType)
    quesFile = '%s/Questions/%s%s_%s_%s_questions.json' % (dataDir, versionType, taskType, dataType, dataSubType)
    imgDir = '%s/Images/%s/' % (dataDir, dataSubType)

    def __init__(self, questionSize, answerSize):
        self.vqa = VQA(self.annFile, self.quesFile)
        self.regex = re.compile('[^A-Za-z0-9 ]+|s$')
        self.questionSize = questionSize
        self.answerSize = answerSize

        questions = self.vqa.questions['questions']
        questions = [question['question'] for question in questions]
        tokens = [token for question in questions for token in self.tokenize(question)]
        self.questionEncoding = self.getMostcommon(tokens, questionSize)

        dataset = self.vqa.dataset
        answers = [self.getAnswer(annotation) for annotation in dataset['annotations']]
        self.answerEncoding = self.getMostcommon(answers, answerSize)

    def getMostcommon(self, tokens, n):
        common = Counter(tokens).most_common(n)
        bag = {answer: idx for idx, (answer, number) in enumerate(common)}

        commonLength = sum([count for key, count in common])
        print('Coverage: ' + str(commonLength/len(tokens)))

        return bag

    def getAnswer(self, ann):
        answers = Counter(answer['answer'] for answer in ann['answers']).most_common(1)
        return answers[0][0]

    def encodeQuestion(self, question):
        tokens = self.tokenize(question)
        encoding = np.zeros((self.questionSize,))
        for token in tokens:
            if token in self.questionEncoding:
                encoding[self.questionEncoding[token]] = 1
        return encoding

    def encodeAnswer(self, answer):
        encoding = np.zeros((self.answerSize,))
        if answer in self.answerEncoding:
            encoding[self.answerEncoding[answer]] = 1
        return encoding

    def tokenize(self, question):
        tokens = [token for token in str.split(question)]
        tokens = [self.regex.sub('', token.lower()) for token in tokens]
        return tokens

    def export(self):
        with open(self.dataDir+"/Temp/questionEncoding.json", 'w') as fp:
            json.dump(self.questionEncoding, fp)
        with open(self.dataDir+"/Temp/answerEncoding.json", 'w') as fp:
            json.dump(self.answerEncoding, fp)

    def encodeAll(self, annIds, model, pageSize):
        questionsEncoded = np.zeros((pageSize, self.questionSize), dtype=np.int8)
        imagesEncoded = np.zeros((pageSize, 2048))
        answersEncoded = np.zeros((pageSize, self.answerSize), dtype=np.int8)
        ids = np.zeros((pageSize,), dtype=np.int32)
        k = 0
        page = 0
        for annId in annIds:
            ann = self.vqa.qa[annId]
            answer = self.getAnswer(ann)
            if answer in self.answerEncoding:
                answersEncoded[k, self.answerEncoding[answer]] = 1
                ids[k] = annId
                question = self.vqa.qqa[annId]['question']
                questionsEncoded[k, :] = self.encodeQuestion(question)

                imgId = self.vqa.qa[annId]['image_id']
                imgPath = self.imgDir+'COCO_' + self.dataSubType + '_' + str(imgId).zfill(12) + '.jpg'
                img = load_img(imgPath, target_size=(299, 299))
                img_array = img_to_array(img)
                imagesEncoded[k, :] = model.predict(np.expand_dims(img_array, axis=0), 1)[0]
                k += 1
                if k >= pageSize:
                    np.save(self.dataDir+"/Encoded/questions_"+str(page), questionsEncoded)
                    np.save(self.dataDir+"/Encoded/images_"+str(page), imagesEncoded)
                    np.save(self.dataDir+"/Encoded/answers_"+str(page), answersEncoded)
                    np.save(self.dataDir+"/Encoded/ids_"+str(page), ids)
                    print("saved "+str(len(ids))+" questions.")
                    page += 1
                    k = 0
        if k > 0:
            np.save(self.dataDir+"/Encoded/questions_"+str(page), questionsEncoded[0:k])
            np.save(self.dataDir+"/Encoded/images_"+str(page), imagesEncoded[0:k])
            np.save(self.dataDir+"/Encoded/answers_"+str(page), answersEncoded[0:k])
            np.save(self.dataDir+"/Encoded/ids_"+str(page), ids[0:k])
            print("saved "+str(len(ids))+" questions.")
