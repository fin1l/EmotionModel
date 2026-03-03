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
with open("03-03-Survey-Responses.csv") as f:
    surveyData = csv.DictReader(f)
    for row in surveyData:
        # Skip rows that failed the attention check
        if row['attention_check'] or (row['greyscale'] == "Yes") or (row['colourblind'] == "Yes"):
            continue
        # Iterate over questions 1-17
        for i in range(1,17):
            totalResponses[i-1].append(getEmotionVector(row["Q"+str(i)]))
# Perform data analysis
responseArray = np.array(totalResponses)
# Get question means
questionMeanVectors = np.mean(responseArray, axis=1)
# Normalise question means
questionNormalisedMeans = questionMeanVectors/np.linalg.norm(questionMeanVectors, axis=1, keepdims=True)
# Dot product for each question works as all vectors are normalised here
questionCosDistances = np.vecdot(questionNormalisedMeans, BASE_RESPONSE_VECTORS)
# Perform dot product between values and intrinsic (Frésnel) means
questionSpreadAngleCos = np.einsum('nmk,nk->nm', responseArray, questionNormalisedMeans)
squaredAngles = np.arccos(questionSpreadAngleCos) ** 2

# Question variances - can't just use built in variance
questionVariances = squaredAngles.mean(axis=1)
# Maximal variance is different for a single emotion or a dyad
MAX_VARIANCE_SINGLE = np.arccos(1/np.sqrt(7)) ** 2
MAX_VARIANCE_DYADIC = np.arccos(np.sqrt(2/7)) ** 2
questionNormalisedVariances = questionVariances
questionNormalisedVariances[:6] /= MAX_VARIANCE_SINGLE
questionNormalisedVariances[6:] /= MAX_VARIANCE_DYADIC
print(MAX_VARIANCE_SINGLE, MAX_VARIANCE_DYADIC)

print(f"Dataset size: {responseArray.shape[1]} responses")
#print(f"Cosine distances for each question: " + "\n".join(f"Q{i}: {questionCosDistances[i-1]}" for i in range(1,17))+"\n")
#print(f"Average vectors for each question: " + "\n".join(f"Q{i}: {questionMeanVectors[i-1]}" for i in range(1,17))+"\n")
#print(f"Normalised variances for each question: " + "\n".join(f"Q{i}: {questionNormalisedVariances[i-1]}" for i in range(1,17))+"\n")

# DATA VISUALISATION

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from adjustText import adjust_text

# Create a DataFrame for easier plotting

df = pd.DataFrame({
    'Question': [f'Q{i}' for i in range(1,17)],
    'Cosine_Distance': questionCosDistances,
    'Variance': questionNormalisedVariances
})

# Quadrant Scatter Plot
accuracyAgreementFigure, scatterAxis = plt.subplots(figsize=(10, 6))
scatterAxis.scatter(df['Cosine_Distance'], df['Variance'], color='dodgerblue', s=100, edgecolors='black', alpha=0.7)
scatterAxis.axvline(0.5, color='gray', linestyle='--', alpha=0.6)
scatterAxis.axhline(0.5, color='gray', linestyle='--', alpha=0.6)
# Label points
texts = []
for i, row in df.iterrows():
    texts.append(scatterAxis.text(row['Cosine_Distance'], row['Variance'], row['Question'], fontsize=9))
scatterAxis.set_title('Consensus vs Accuracy')
scatterAxis.set_xlabel('Cosine Distance')
scatterAxis.set_ylabel('Normalised Variance')
scatterAxis.set_xlim(0, 1)
scatterAxis.set_ylim(0, 1)
adjust_text(texts,
    force_points=0.2,
    force_text=0.2,
    expand_points=(1.2, 1.2))
ticks = [0.0, 0.25, 0.5, 0.75, 1.0]
scatterAxis.set_xticks(ticks)
scatterAxis.set_yticks(ticks)
scatterAxis.grid(True, linestyle=':', alpha=0.6)

# Sorted bar chart view - rank by cos distance and colour by variance
df_sorted = df.sort_values('Cosine_Distance', ascending=True)
fig2, ax2 = plt.subplots(figsize=(10, 8))
ax2.set_xlim(0, 1)
ax2.set_xticks(ticks)
# Colour mapping
norm = mcolors.Normalize(vmin=0, vmax=1)
cmap = cm.RdYlGn_r

# Horizontal Bar Chart
bars = ax2.barh(df_sorted['Question'], df_sorted['Cosine_Distance'], color=cmap(norm(df_sorted['Variance'])))
# Colour bar for context
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
colourBar = plt.colorbar(sm, ax=ax2)
colourBar.set_label('Normalised Variance (Red=Disagreement, Green=Consensus)')

ax2.set_xlabel('Cosine Distance')
ax2.set_title('Question Accuracy Ranking (with Consensus Colouring)')
plt.show()
