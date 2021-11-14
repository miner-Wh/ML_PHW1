from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
from sklearn import preprocessing
import warnings ; warnings.filterwarnings('ignore')
import seaborn as sns
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score, StratifiedKFold 
skf = StratifiedKFold(n_splits=10) 
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import sys
################
#make result table
score_sample = {'Scaler':["Sample"], 'Encoder':["Sample"], 'Model':["Sample"],'Best_para':["Sample"], "Score":[1]}
score_results = pd.DataFrame(score_sample)



#for scale and encorde
class PreprocessPipeline(): 
    def __init__(self, num_process, cat_process, verbose=False): 
        #super(PreprocessPipeline, self).__init__() 
        self.num_process = num_process 
        self.cat_process = cat_process 
        #for each type
        if num_process == 'standard': 
            self.scaler = preprocessing.StandardScaler() 
        elif num_process == 'minmax': 
            self.scaler = preprocessing.MinMaxScaler() 
        elif num_process == 'maxabs': 
            self.scaler = preprocessing.MaxAbsScaler() 
        elif num_process == 'robust': 
            self.scaler = preprocessing.RobustScaler() 
        else: 
            raise ValueError("Supported 'num_process' : 'standard','minmax','maxabs','robust'")   
        if cat_process == 'onehot': 
            self.encoder = preprocessing.OneHotEncoder(sparse=False, handle_unknown='ignore')  
        elif cat_process == 'ordinal': 
            self.encoder = preprocessing.OrdinalEncoder() 
        else: 
            raise ValueError("Supported 'cat_process' : 'onehot', ordinal'") 

        self.verbose=verbose 
        
        #do Preprocess
    def process(self, X, Xt): 
        X_cats = X.select_dtypes(np.object).copy() 
        X_nums = X.select_dtypes(exclude=np.object).copy() 
        Xt_cats = Xt.select_dtypes(np.object).copy() 
        Xt_nums = Xt.select_dtypes(exclude=np.object).copy() 

        if self.verbose: 
            print(f"Categorica Colums : {list(X_cats)}") 
            print(f"Numeric Columns : {list(X_nums)}") 

        if self.verbose: 
            print(f"Categorical cols process method : {self.cat_process.upper()}") 

        X_cats = self.encoder.fit_transform(X_cats) 
        Xt_cats = self.encoder.transform(Xt_cats) 

        if self.verbose: 
            print(f"Numeric columns process method : {self.num_process.upper()}") 
        X_nums = self.scaler.fit_transform(X_nums) 
        Xt_nums = self.scaler.transform(Xt_nums) 

        X_processed = np.concatenate([X_nums, X_cats], axis=-1) 
        Xt_processed = np.concatenate([Xt_nums, Xt_cats], axis=-1) 
     
        return X_processed, Xt_processed 

# do process on I want 
class AutoProcess():
    def __init__(self, verbose=False):
        
        self.pp = PreprocessPipeline
        self.verbose= verbose
    
    def run(self, X, Y, Xt, Yt):
        methods = []
        scores = []
        print(X.shape, Xt.shape)
        
        for num_process in ['standard','robust','minmax','maxabs']:
            for cat_process in ['onehot','ordinal']:
                if self.verbose:
                    print("\n------------------------------------------------------\n")
                    print(f"Numeric Process : {num_process}")
                    print(f"Categorical Process : {cat_process}")
                methods.append([num_process, cat_process])

                pipeline = self.pp(num_process=num_process, cat_process=cat_process)
                
                X_processed, Xt_processed = pipeline.process(X, Xt)

                #Classifier part
                for model in ['gini','entropy','svc']:
                    if self.verbose:
                        print(f"\nClassifier model: {model}")

                    if model =='gini': 
                        param_grid = {'max_depth' : [3,5,7,10],
                                      "min_samples_leaf":[1, 2, 3, 4, 5, 6, 7, 8, 9],
                                      "min_samples_split":[2, 3, 4, 5, 6, 7, 8, 9, 10]}
                        clf = DecisionTreeClassifier()
                    elif model =='svc':
                        param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100],
                                     'gamma': [0.001, 0.01, 0.1, 1, 10, 100] }
                        clf = SVC()
                    else:
                        param_grid = {'max_depth' : [3,5,7,10],
                                      "min_samples_leaf":[1, 2, 3, 4, 5, 6, 7, 8, 9],
                                      "min_samples_split":[2, 3, 4, 5, 6, 7, 8, 9, 10]}
                        clf = DecisionTreeClassifier(criterion = "entropy")

                    grid_search = GridSearchCV(clf,param_grid,cv=5)
                    grid_search.fit(X_processed,Y)
                    score_results.loc[len(score_results)] = [num_process, cat_process, model, grid_search.best_params_, grid_search.best_score_]

                    clf = clf.fit(X_processed, Y)  
                    predict = clf.predict(Xt_processed)
                    score1 = accuracy_score(Yt, predict)
                    #print("Score: ", score1)

                    score = cross_val_score(clf, Xt_processed, Yt, cv=kfold, n_jobs=1, scoring='accuracy')
                    score2 = np.mean(score)
                    #print("Score with using kfold: ", score2)

                    score_results.loc[len(score_results)] = [num_process, cat_process, model, np.NaN ,score1]
                    score_results.loc[len(score_results)] = [num_process, cat_process, model+" with kfold",np.NaN, score2]
                
                #logistic Regression part
                for model in ['logistic']:
                    if self.verbose:
                        print(f"\nRegression model: {model}")
                    
                    param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100],
                                  'penalty': ['l1', 'l2']}
                    grid_search =GridSearchCV(LogisticRegression(), param_grid, cv=5)
                    grid_search.fit(X_processed, Y)
                    
                    #lr = LogisticRegression().fit(X_processed, Y)
                    #predict = lr.predict(Xt_processed)
                    #score = round(accuracy_score(Yt, predict.round())*100, 2)
                    score_results.loc[len(score_results)] = [num_process, cat_process, model, grid_search.best_params_, grid_search.best_score_]

        return


kfold = KFold(5, True, 1)




df = pd.read_csv('C:\Users\Minner\Documents\VSC_workS\ML_PHW1\breast-cancer-wisconsin.data', sep=',',names=['ID','CT','UC Size','UC Shape','MA','SECS','BN','BC','NN','Mitoses','Class'],header=None)
#print(df)

#check null
print(df.isnull().sum())
#check data type
print(df.dtypes)
df=df[df.BN != '?']
df=df.astype({'BN':int})
##print(df.dtypes)
df = df.drop(['ID'],axis =1)
print(df)

#Separate taget and feature
X =df.iloc[:,0:9]
Y = df.iloc[:,[9]]
print(X)
print(Y)

#split train and test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=0)


autoprocess = AutoProcess(verbose=True)
autoprocess.run(X_train, Y_train, X_test, Y_test)


pd.set_option('display.max_row', 100)
print(score_results.sort_values(by=['Score'], axis=0,ascending=False))
sys.stdout = open('score result.txt', 'w')

print(score_results.sort_values(by=['Score'], axis=0,ascending=False))

sys.stdout.close()