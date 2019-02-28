class VQAConfig:
    def __init__(self, 
        batchSize=512, 
        imageType=None,       
        imageFeaturemapSize = 24,         
        imageFeatureChannels = 1536,
        gloveName='glove.6B.100d',
        gloveSize=100,
        testName='test',
        augmentations=None,
        dropout=True,
        gatedTanh=False
        ):

        self.batchSize = batchSize
        self.imageType = imageType
        self.imageFeaturemapSize = imageFeaturemapSize
        self.imageFeatureChannels = imageFeatureChannels
        self.gloveName = gloveName
        self.gloveSize = gloveSize
        self.testName = testName
        self.augmentations = augmentations
        self.dropout=dropout
        self.gatedTanh=gatedTanh
    
    def __str__(self):
        opts = vars(self)
        return '\n'.join('%s: %s' % item for item in opts.items())