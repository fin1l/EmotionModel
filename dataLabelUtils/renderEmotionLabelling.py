import PySimpleGUI as sg
import json
import os

DEFAULT_DATA_FOLDER = "" 

EMOTION_LABELS = [
    "anger",
    "disgust",
    "fear",
    "joy",
    "sadness",
    "surprise",
    "neutral",
]

def loadData(dataPath):
    JSONPath = os.path.join(dataPath, "labels.json")
    imagesPath = os.path.join(dataPath, "images")
    
    if not os.path.exists(JSONPath):
        sg.popup_error(f"Error: labels not found in {dataPath}")
        return None, None
        
    with open(JSONPath, 'r') as f:
        data = json.load(f)
        
    return data, imagesPath, JSONPath

def getUnlabeledInd(data, start_index=0):
    for index in range(start_index, len(data)):
        if not data[index].get("labels") or len(data[index]["labels"]) == 0:
            return index
    return None

def createMainUI():
    sliders = []
    for index, label in enumerate(EMOTION_LABELS):
        sliders.append([
            sg.Text(label, size=(20, 1)),
            sg.Slider(range=(0.0, 1.0), default_value=0.0, resolution=0.01,
                orientation='h', size=(40, 15), key=f"-SLIDER-{index}-")
        ])
    layout = [
        [sg.Text("Image:", size=(5,1)), sg.Text("", key="-IMAGE-ID-", size=(60,1))],
        [sg.Image(key="-IMAGE-DISPLAY-", size=(800, 600), background_color='black')],
        [sg.HorizontalSeparator()], *sliders, [sg.HorizontalSeparator()],
        [sg.Button("Next", key="-SAVE-", size=(15, 2)), sg.Button("Exit", size=(10, 2)),
         sg.Text("", key="-STATUS-", size=(40, 1))]
    ]
    return sg.Window("Data Labelling", layout, finalize=True, resizable=True)

def loadImage(window, imagePath):
    try:
        window["-IMAGE-DISPLAY-"].update(source=imagePath)
    except Exception as e:
        print(f"Error loading image {imagePath}: {e}")
        window["-IMAGE-DISPLAY-"].update(data=None)

def main():
    dataFolder = DEFAULT_DATA_FOLDER
    if not dataFolder:
        dataFolder = sg.popup_get_folder("Select the folder containing labels.json", "Select Data Folder")
    if not dataFolder:
        return
    data, imagesPath, JSONPath = loadData(dataFolder)
    if data is None:
        return
    current_index = getUnlabeledInd(data, 0)
    if current_index is None:
        sg.popup("Labelling done")
        return
    window = createMainUI()

    entry = data[current_index]
    imagePath = os.path.join(imagesPath, entry["image_id"])
    window["-IMAGE-ID-"].update(entry["image_id"])
    loadImage(window, imagePath)

    while True:
        event, values = window.read()
        
        if event == sg.WIN_CLOSED or event == "Exit":
            break
            
        if event == "-SAVE-":
            labels = {}
            for i, key in enumerate(EMOTION_LABELS):
                slider_key = f"-SLIDER-{i}-"
                labels[key] = values[slider_key]
            
            # save labels + save JSON file
            data[current_index]["labels"] = labels
            try:
                with open(JSONPath, 'w') as f:
                    json.dump(data, f, indent=2)
                window["-STATUS-"].update(f"Saved {data[current_index]['image_id']}")
            except Exception as e:
                window["-STATUS-"].update(f"Error saving: {e}")
                continue
            current_index = getUnlabeledInd(data, current_index + 1)
            if current_index is None:
                sg.popup("Labelling done")
                break
            
            for i in range(len(EMOTION_LABELS)):
                window[f"-SLIDER-{i}-"].update(0.0)
            
            entry = data[current_index]
            imagePath = os.path.join(imagesPath, entry["image_id"])
            window["-IMAGE-ID-"].update(entry["image_id"])
            loadImage(window, imagePath)

    window.close()

if __name__ == "__main__":
    main()