import matplotlib.pyplot as plt
import numpy as np

# Code is repeated from regressionModel.py
def plotKFoldResults(architectureResults):
    plt.figure(figsize=(10, 6))
    modelNames = list(architectureResults.keys())
    # Losses for each fold
    modelLosses = list(architectureResults.values())
    # Boxplot to show variance across folds for each architecture
    bplot = plt.boxplot(modelLosses, tick_labels=modelNames, patch_artist=True)
    print([datum.get_ydata() for datum in bplot['medians']])
    for patch in bplot['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_edgecolor('black')
    plt.title('Architecture Performance (K-Fold Cross Validation)')
    plt.ylabel('MSE Loss')
    plt.xlabel('Model Architecture')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig('kFoldArchitectureComparison.png')
    plt.show()

def plotDropoutConfigurations(configurationResults):
    configNames = list(configurationResults.keys())
    avgLosses = list(configurationResults.values())
    
    plt.figure(figsize=(10, 6))
    # Use bar chart instead
    plt.bar(configNames, avgLosses, color='skyblue', edgecolor='black')
    plt.title('Dropout Strategy Performance (K-Fold Cross Validation)')
    plt.ylabel('Average MSE Loss')
    plt.xlabel('Dropout Configuration')
    plt.xticks(rotation=15)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig('dropoutStrategyComparison.png')
    plt.tight_layout()
    plt.show()

# Results ported from regressionModel.py for plotting
# Stops me having to run the whole training process to update formatting
architectureResults = {
    "Base Model": [0.01099,0.01231,0.01313,0.01135,0.01220,0.01265,0.01241,0.01434],
    "Deep Model": [0.00948,0.00983,0.01064,0.00936,0.00851,0.01047,0.00988,0.01087],
    "Deep Model 2": [0.00948,0.01023,0.01029,0.00847,0.00894,0.01056,0.00965,0.01045],
    "Refined Model": [0.00862,0.00883,0.01097,0.00834,0.00824,0.00957,0.00870,0.00994]
}
print(np.quantile(architectureResults["Refined Model"], [0.25, 0.5, 0.75]))
dropoutRawResults = {
    "Aggressive Constant": [0.00930, 0.00943, 0.00871, 0.00934, 0.00932],
    "Moderate Constant": [0.00896, 0.00948, 0.00853, 0.00944, 0.00950],
    "Aggressive Taper": [0.00880, 0.00980, 0.00810, 0.00965, 0.00928],
    "Gentle Taper": [0.00954, 0.00988, 0.00930, 0.00965, 0.00897],
    "Baseline (Current)": [0.00862, 0.00947, 0.00790, 0.00910, 0.00959]
}
dropoutResults = {name: sum(losses)/len(losses) for name, losses in dropoutRawResults.items()}
plotKFoldResults(architectureResults)
plotDropoutConfigurations(dropoutResults)
