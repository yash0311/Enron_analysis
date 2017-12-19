#!/usr/bin/python
import sys
import pickle
from sklearn import tree
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data,test_classifier
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot
from sklearn.cross_validation import train_test_split

sys.path.append("../tools/")
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

######## OUTLIERS REMOVAL START #######

outliers = []    
for key in data_dict:
    if(data_dict[key]['salary']=='NaN' or data_dict[key]['salary']>600000 or data_dict[key]['salary']<1000):
        if(data_dict[key]['salary']!='NaN'):
            outliers.append((key,int(data_dict[key]['salary'])))
        continue
    salary = data_dict[key]['salary']
    bonus = data_dict[key]['bonus']
    matplotlib.pyplot.scatter( salary, bonus )


matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()

for x in outliers:
    print(x)
    data_dict.pop(x[0],0)
########  OUTLIERS REMOVAL END #######


####### NEW ENRON FEATURE CREATION #####

def compute_fraction(poi_messages, all_messages):
    if poi_messages=='NaN' or all_messages=='NaN':
        return 0.
    else:
        return float(poi_messages)/float(all_messages)


for key in data_dict:
    data_dict[key]["fraction_from_poi"]=compute_fraction(data_dict[key]['from_poi_to_this_person'], data_dict[key]['to_messages'])
    data_dict[key]["fraction_to_poi"]=compute_fraction(data_dict[key]['from_this_person_to_poi'], data_dict[key]['from_messages'])

####### NEW ENRON FEATURE END #####


######  SELECTION OF BEST FEATURES FOR ANALYSIS #######
my_dataset = data_dict
features_list=['poi','shared_receipt_with_poi','total_stock_value','fraction_to_poi','fraction_from_poi','expenses','bonus']


data = featureFormat(my_dataset, features_list,remove_NaN=True, sort_keys = False)
labels, features = targetFeatureSplit(data)

k=4
k_best = SelectKBest(k=k)
k_best.fit(features, labels)
scores = k_best.scores_
print(scores)
features_list=['poi','shared_receipt_with_poi','total_stock_value','fraction_to_poi','bonus']

####### FEATURE SELECTION END #######


####### FEATURE SCALING START #######
data = featureFormat(my_dataset, features_list,remove_NaN=True, sort_keys = False)
labels, features = targetFeatureSplit(data)

for i in range(len(features_list)-1):
    tmp =[]
    k=0
    for x in features:
        tmp.append(float(x[i]))
    tmp = MinMaxScaler().fit_transform(tmp)
    for x in features:
        x[i]=tmp[k]
        k = k + 1
####### FEATURE SCALING END #######


####### TRAINING & TESTING USING CLASSIFIER ########
clf=tree.DecisionTreeClassifier()
test_classifier(clf,my_dataset,features_list)

features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

clf.fit(features_train,labels_train)
pred=clf.predict(features_test)

print accuracy_score(labels_test,pred)
print(recall_score(labels_test, pred))
####### TRAINING & TESTING USING CLASSIFIER END ########


##### DUMP CLASSIFIER SO ANYONE CAN SEE RESULT #####
dump_classifier_and_data(clf, my_dataset, features_list)
###### DUMP END ######