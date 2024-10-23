
from sklearn.svm import SVC


class WeakClassifier():

    def __init__(self, params):
        self.model = SVC(**params)
        self.trained = False
    
    def fit(self, x_train, y_train): 
        self.model.fit(x_train, y_train)
        self.trained = True
        return self
    
    
    def predict(self, x):
        if not self.trained:
            raise Exception('Model not trained')
        
        y_pred_val = self.model.predict(x)
        y_pred_scores = self.model.predict_proba(x)
        return y_pred_val, y_pred_scores[:,1]

