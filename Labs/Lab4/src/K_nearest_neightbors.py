import numpy as np

dtype = [('distance', float), ('target', float)]

def classifier_Knn_numpy(x_train,x_new,targets,K):
    distances = []
    counter1 = 0
    counter2 = 0
    for i in range(0,len(x_train)):
        distances.append((np.linalg.norm(x_new-x_train[i]),targets[i]))
    distances=np.array(distances,dtype=dtype)
    distances=np.sort(distances,order='distance')  # plus petites en premier
    for j in range(0,K):
        if distances[j][1]==1.:
            counter1+=1
        if distances[j][1]==2.:
            counter2+=1
        if counter1+counter2==K:
            break
    if counter1>counter2:
        return 1.
    else:
        return 2.

def testWithTrainOnTest(x_train,X_new,class_train,K):
    target_prediction_test = []
    for x in X_new:
        target_prediction_test.append(classifier_Knn_numpy(x_train,x_new=x,targets=class_train,K=K))
    return target_prediction_test


#Compute error rate
def computeError(target_prediction_test,class_test):
    return sum(abs(target_prediction_test-class_test))/len(class_test)