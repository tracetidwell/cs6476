

def classifyAllActions(hu_moments, labels):

	n_classes = len(set(labels))
	confusion_matrix = np.zeros((n_classes, n_classes))

	for i, (hu_moment, label) in enumerate(zip(hu_moments, labels)):
	    
	    x_train = np.concatenate([hu_moments[:i], hu_moments[i+1:]])
	    y_train = np.concatenate([labels[:i], labels[i+1:]])
	    y_hat = predictAction(hu_moment, x_train, y_train)
	    confusion_matrix[label[0], y_hat] += 1

	recognition_rate = np.sum([confusion_matrix[i, i] for i in range(5)]) / np.sum(confusion_matrix)