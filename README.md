# About
This repository is for a Blender tool to train a regression model to map emotion to a set of scene parameters in Blender (by adjusting the camera and lighting)

# Set up
The lookdev scene (`inference_look_dev.blend`) included already has the tool installed as an add-on, but the `EmotionGenerationAddon.zip` file can be directly installed into any Blender project. The model weights file (`emotionInferenceModel.pth`) is also required by the tool.

A three-point lighting structure is required for the tool to function best, as is demonstrated in the lookdev scene.

## Use
Once the model weights file has been selected through the file browser in the add-on and all of the lights have been linked, the input emotion can be chosen. This is either done using the slider mode or through sentiment extraction from text with the input mode togglable by a button.

# Acknowledgements
The text extraction section of this project uses the [Emotion English DistilRoBERTa-base](https://huggingface.co/j-hartmann/emotion-english-distilroberta-base) model available on Hugging Face.
