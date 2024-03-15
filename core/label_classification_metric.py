
class LabelClassificationMetric:

    human_label = None
    label = None
    correct = 0
    incorrect = 0
    

    def __init__(self, label, human_label) -> None:
        self.label = label
        self.human_label = human_label

    def addCorrect(self):
        self.correct += 1

    def addIncorrect(self):
        self.incorrect += 1
    
    def total(self):
        return self.incorrect + self.correct
    
    def ratio(self):
        if self.incorrect == 0:
            return -1
        else:
            return self.correct/self.incorrect
    
    def percent_correct(self):
        if self.total() == 0:
            return -1
        else:
            return (self.correct/self.total()) * 100
    
    def percent_incorrect(self):
        if self.total() == 0:
            return -1
        else:
            return (self.incorrect/self.total()) * 100
    
    