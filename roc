# Convert one-hot encoded labels back to integer labels
y_test_labels = np.argmax(y_test, axis=1)

# Calculate the ROC curve and AUC for each class using the one-vs-rest approach
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(3):
    # Binarize the true labels and the predicted probabilities for the current class
    y_test_bin = (y_test_labels == i).astype(int)
    y_prob_bin = y_prob[:, i]

    fpr[i], tpr[i], _ = roc_curve(y_test_bin, y_prob_bin)
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot the ROC curves
plt.figure()
colors = ['aqua', 'darkorange', 'cornflowerblue']
for i, color in enumerate(colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f'Class {i} (area = {roc_auc[i]:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic for Multi-Class')
plt.legend(loc='lower right')
plt.show()
