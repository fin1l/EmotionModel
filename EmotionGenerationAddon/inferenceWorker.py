import sys
import os
import json
try:
    import torch
    import torch.nn as nn
    import transformers
except ImportError as e:
    # Print JSON error so Blender can read it
    print(json.dumps({"status": "error", "message": f"Worker Import Failed: {e}"}))
    sys.exit(1)
# Disable verbose logging (as console output is read by modelUtils)
transformers.logging.set_verbosity_error()

class ImprovedDeepEmotionModel(nn.Module):
    def __init__(self):
        super(ImprovedDeepEmotionModel, self).__init__()
        self.modelLayers = nn.Sequential(
            nn.Linear(7, 256), nn.BatchNorm1d(256), nn.ELU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ELU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ELU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.ELU(),
            nn.Linear(32, 6), nn.Sigmoid()
        )
    
    def forward(self, inputTensor):
        return self.modelLayers(inputTensor)

EMOTION_LABEL_INDICES = {label:index for index,label in enumerate(("anger","disgust","fear","joy","sadness","surprise","neutral"))}

TEXT_MODEL_SUBDIR = "textModel"

EMOTION_MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"

def loadEmotionClassifier(path, modelName):
    modelConfig = os.path.join(path, "config.json")
    if os.path.exists(path) and os.path.exists(modelConfig):
        modelSource = path
        shouldSave = False
    else:
        modelSource = modelName
        shouldSave = True
    classifier = transformers.pipeline(
        "text-classification", 
        model=modelSource,
        tokenizer=modelSource,
        top_k=None
    )
    # Save the model if it was loaded for the first time
    if shouldSave:
        classifier.model.save_pretrained(path)
        classifier.tokenizer.save_pretrained(path)
    return classifier

def getEmotionClassifierPath(modelPath):
    modelDirectory = os.path.dirname(modelPath)
    textModelPath = os.path.join(modelDirectory, TEXT_MODEL_SUBDIR)
    os.makedirs(textModelPath, exist_ok=True)
    return textModelPath

if __name__ == "__main__":
    try:
        inputs = []
        if len(sys.argv) == 3:
            #classifier = transformers.pipeline("text-classification", model=EMOTION_MODEL_NAME, top_k=None)
            emotionClassifierPath = getEmotionClassifierPath(sys.argv[1])
            classifier = loadEmotionClassifier(emotionClassifierPath, EMOTION_MODEL_NAME)
            # Comes nested in another array - need to get first element
            textEmotions = classifier(sys.argv[2])[0]
            inputs = [0 for _ in range(7)]
            for labelValuePair in textEmotions:
                inputs[EMOTION_LABEL_INDICES[labelValuePair["label"]]] = labelValuePair["score"]
        elif len(sys.argv) ==9:
            inputs = [float(x) for x in sys.argv[2:]]
        else:
            raise ValueError(f"Expected 9 arguments, found {len(sys.argv)}")
        # Unpack arguments
        modelPath = sys.argv[1]
        
        # Use CUDA if possible
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        model = ImprovedDeepEmotionModel().to(device)
        
        if not os.path.exists(modelPath):
            raise FileNotFoundError(f"Model file not found: {modelPath}")

        # Load model weights
        model.load_state_dict(torch.load(modelPath, map_location=device))
        model.eval()
        
        # Model inference
        inputTensor = torch.tensor([inputs], dtype=torch.float32).to(device)
        with torch.no_grad():
            output = model(inputTensor)
            
        # Return result in JSON format
        result = output.tolist()[0]
        print(json.dumps({"status": "success", "data": result}))
        
    except Exception as e:
        print(json.dumps({"status": "error", "message": str(e)}))
