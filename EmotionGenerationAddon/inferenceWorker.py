import sys
import os
import json
try:
    import torch
    import torch.nn as nn
except ImportError as e:
    # Print JSON error so Blender can read it
    print(json.dumps({"status": "error", "message": f"Worker Import Failed: {e}"}))
    sys.exit(1)

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

if __name__ == "__main__":
    try:
        if len(sys.argv) < 9:
            raise ValueError(f"Expected 9 arguments, found {len(sys.argv)}")
        # Unpack arguments
        modelPath = sys.argv[1]
        inputs = [float(x) for x in sys.argv[2:]]
        
        # Use CUDA if possible
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
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
