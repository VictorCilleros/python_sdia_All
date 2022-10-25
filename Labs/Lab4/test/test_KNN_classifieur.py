import unittest
from ..src.K_nearest_neightbors import classifier_Knn_numpy,testWithTrainOnTest,computeError
import numpy as np

class Test_Knn(unittest.TestCase):

    def test_result_classifieur(self):
        train = np.loadtxt('Lab4/data/synth_train.txt')  #...,delimiter=',') if there are ',' as delimiters
        class_train = train[:,0]
        x_train = train[:,1:]
        test = np.loadtxt('Lab4/data/synth_train.txt')
        x_test = test[:,1:]
        classement = classifier_Knn_numpy(x_train,x_new=x_test[0],targets=class_train,K=3)
        self.assertIsInstance(classement, float)
        self.assertEqual((classement==1.) | (classement==2.), True)


    def test_prediction_classifieur(self):
        train = np.loadtxt('Lab4/data/synth_train.txt')  #...,delimiter=',') if there are ',' as delimiters
        class_train = train[:,0]
        x_train = train[:,1:]
        test = np.loadtxt('Lab4/data/synth_train.txt')
        x_test = test[:,1:]
        predictions=testWithTrainOnTest(x_train,x_test,class_train,3)
        self.assertIsInstance(predictions, list)
        self.assertEqual(len(predictions), len(test))

    def test_error_rate(self):
        train = np.loadtxt('Lab4/data/synth_train.txt')  #...,delimiter=',') if there are ',' as delimiters
        class_train = train[:,0]
        x_train = train[:,1:]
        test = np.loadtxt('Lab4/data/synth_train.txt')
        class_test=test[:,0]
        x_test = test[:,1:]
        error=computeError(testWithTrainOnTest(x_train,x_test,class_train,3),class_test)
        self.assertIsInstance(error, float)
        self.assertEqual(error<=1. and error>=0., True)