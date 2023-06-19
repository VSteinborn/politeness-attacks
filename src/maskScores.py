from torch.utils.data import DataLoader
from util.io import readArgs, loadFiles, dfSave, loadModel, getDevice
from util.formating import buildSentences, getMaxTokenLength, SentenceDataset, insertMaskInSentence, getTokenIDOfExistingSearchTerms
from util.evaluation import evaluateDataset


def main():
    # load sentences
    args = readArgs()
    sentenceCombinations = loadFiles(args)

    dfLines=[]
    for modelName in sentenceCombinations["models"].values():
        # make sentences
        sentenceData = buildSentences(sentenceCombinations, templateKey=args.template, applicationKey=args.application)

        # load model
        device = getDevice(args.batchsize)
        model, tokenizer = loadModel(modelName, device, applicationKey=args.application)

        # Tokenization and formatting
        sentenceData["sentence"]=sentenceData.apply(insertMaskInSentence, tokenizer=tokenizer, axis=1)
        maxInputLength= getMaxTokenLength(sentenceData["sentence"], tokenizer)
        dataset = SentenceDataset(tokenizer, sentenceData, maxInputLength, applicationKey=args.application)
        dataloader = DataLoader(dataset, batch_size=args.batchsize, shuffle=False)

        # Evaluation
        existingSearchTerms=getTokenIDOfExistingSearchTerms(tokenizer, sentenceCombinations["genderTerms"])
        runsData = evaluateDataset(dataloader, model, device, existingSearchTerms=existingSearchTerms, applicationKey=args.application)
        dfLines.extend(runsData)

    dfSave(dfLines, runId=args.id)


if __name__ == "__main__":
    main()
    