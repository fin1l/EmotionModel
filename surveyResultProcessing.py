import csv
import math
import numpy as np
EMOTION_INDICES = {emotion: index for index, emotion in
                   enumerate(["Anger", "Disgust", "Fear", "Joy", "Sadness", "Surprise", "Neutral"])}
ROOT_TWO_WEIGHTING = 1.0 / math.sqrt(2)
def getEmotionVector(inputString):
    emotionVector = np.zeros(7, dtype=float)
    emotionList = inputString.split(", ")
    # Assumes that responses will only ever have 1 or 2 emotions (enforced by tally.so)
    normalisedScaling = 1.0 if len(emotionList) < 2 else ROOT_TWO_WEIGHTING
    for emotion in emotionList:
        emotionVector[EMOTION_INDICES[emotion]] = normalisedScaling
    return emotionVector

baseResponseStrings = ["Anger", "Joy", "Fear", "Surprise", "Sadness",
                       "Disgust", "Joy, Surprise", "Disgust, Fear",
                       "Anger, Fear", "Sadness, Surprise", "Anger, Surprise",
                       "Disgust, Sadness", "Fear, Surprise", "Anger, Disgust",
                       "Joy, Sadness", "Fear, Sadness"]
BASE_RESPONSE_VECTORS = np.array([getEmotionVector(response) for response in baseResponseStrings])

totalResponses = [[getEmotionVector(response)] for response in baseResponseStrings]#[[] for _ in range(16)]
with open("mock.csv") as f:
    surveyData = csv.DictReader(f)
    for row in surveyData:
        # Skip rows that failed the attention check
        if row['attention_check']:
            continue
        # Iterate over questions 1-17
        for i in range(1,17):
            totalResponses[i-1].append(getEmotionVector(row["Q"+str(i)]))
# Perform data analysis
responseArray = np.array(totalResponses)
# Get question means
questionMeanVectors = np.mean(responseArray, axis=1)
# Dot product for each question works as all vectors are normalised here
questionCosDistances = np.vecdot(questionMeanVectors, BASE_RESPONSE_VECTORS)
# Question variances - can't just use built in variance
questionVectorDistances = np.linalg.norm(responseArray - questionMeanVectors[:, None, :], axis=2)
questionVariances = questionVectorDistances.mean(axis=1)

print(f"Cosine distances for each question: " + "\n".join(f"Q{i}: {questionCosDistances[i-1]}" for i in range(1,17))+"\n")
print(f"Average vectors for each question: " + "\n".join(f"Q{i}: {questionMeanVectors[i-1]}" for i in range(1,17))+"\n")
print(f"Variances for each question: " + "\n".join(f"Q{i}: {questionVariances[i-1]}" for i in range(1,17))+"\n")
