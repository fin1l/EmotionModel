import sys
import json
import math
import csv

if len(sys.argv) > 1:
    inputJSONFilePath = sys.argv[1]
else:
    inputJSONFilePath = "./EmotionRenders/labels.json"
if len(sys.argv) > 2:
    outputCSVFilePath = sys.argv[2]
else:
    outputCSVFilePath = "./datasets/trainingData.csv"

LABEL_NAMES = ["anger","disgust","fear","joy","sadness","surprise","neutral"]
# Load input data
with open(inputJSONFilePath, 'r') as f:
    rawData = json.load(f)

# Calculate initial min-max scaling values
labelMins, labelMaxes = {emotion:1 for emotion in LABEL_NAMES}, {emotion:0 for emotion in LABEL_NAMES}
for entry in rawData:
    labels = entry["labels"]
    for emotion in labels:
        if labels[emotion] < labelMins[emotion]:
            labelMins[emotion] = labels[emotion]
        if labels[emotion] > labelMaxes[emotion]:
            labelMaxes[emotion] = labels[emotion]

print(labelMins, labelMaxes)

# Apply feature normalisation + min-max scaling
PARAMETER_MIN_MAX = {
    "hueSin": [-1, 1],
    "hueCos": [-1, 1],
    "saturation": [0.2, 1],
    "light_energy": [math.log(50), math.log(3000)],
    "grain": [0, 0.4],
    "fov": [25, 100]
}

DATA_ORDERING = ["anger","disgust","fear","joy","sadness","surprise","neutral","hueSin","hueCos","saturation","light_energy","grain","fov"]
CSV_DATA_INDICES = {DATA_ORDERING[index]: index for index in range(len(DATA_ORDERING))}
CSVRows = []
for entry in rawData:
    currentCSVRow = [0 for _ in range(len(DATA_ORDERING))]
    features = entry["parameters"]
    for feature in features:
        if feature == "light_energy":
            featureValue = math.log(features[feature])
        else:
            featureValue = features[feature]
        newValue = (featureValue - PARAMETER_MIN_MAX[feature][0]) / (PARAMETER_MIN_MAX[feature][1]-PARAMETER_MIN_MAX[feature][0])
        currentCSVRow[CSV_DATA_INDICES[feature]] = newValue
    labels = entry["labels"]
    for emotion in labels:
        newValue = (labels[emotion] - labelMins[emotion]) / (labelMaxes[emotion] - labelMins[emotion])
        currentCSVRow[CSV_DATA_INDICES[emotion]] = newValue
    CSVRows.append(currentCSVRow)

# Write data to the CSV output file
CSV_HEADERS = ["anger","disgust","fear","joy","sadness","surprise","neutral","hueSin","hueCos","saturation","log_light_energy","grain","fov"]
with open(outputCSVFilePath, 'w', newline="") as csvfile:
    fileWriter = csv.writer(csvfile, delimiter=",")
    fileWriter.writerow(CSV_HEADERS)
    fileWriter.writerows(CSVRows)
