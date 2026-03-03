def plotKFoldResults(architectureResults):
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 6))
    modelNames = list(architectureResults.keys())
    # losses for each fold
    modelLosses = list(architectureResults.values())
    # boxplot to show variance across folds for each architecture
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

# Results ported from regressionModel.py for boxplotting
# Stops me having to run the whole training process to update formatting
results = {
    "Base Model": [0.01099,0.01231,0.01313,0.01135,0.01220,0.01265,0.01241,0.01434],
    "Deep Model": [0.00948,0.00983,0.01064,0.00936,0.00851,0.01047,0.00988,0.01087],
    "Deep Model 2": [0.00948,0.01023,0.01029,0.00847,0.00894,0.01056,0.00965,0.01045],
    "Refined Model": [0.00862,0.00883,0.01097,0.00834,0.00824,0.00957,0.00870,0.00994]
}
for k in results:
    print(f"{k}: {len(results[k])}")
plotKFoldResults(results)
