

def accuracy(labels, predictions):
    assert len(labels) == len(predictions)

    correct = 0
    for label, prediction in zip(labels, predictions):
        if label == prediction:
            correct += 1

    score = correct / len(labels)
    return score

                

    

