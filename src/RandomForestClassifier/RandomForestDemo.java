/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package randomforestdemo;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.util.Random;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;

/**
 *
 * @author HP
 */
public class RandomForestDemo {


      public static void main(String[] args) throws Exception {
       
         
        
         
         
          BufferedReader inputReader = null;
         
         
              inputReader = new BufferedReader(new FileReader("iris.arff"));
              
              Instances train = new Instances(inputReader);
              train.setClassIndex(train.numAttributes()-1);
              
              inputReader.close();
              
              NaiveBayes nb = new NaiveBayes();
              nb.buildClassifier(train);
              
              Evaluation eval = new Evaluation(train);
              eval.crossValidateModel(nb, train, 10, new Random(1));
              System.out.println(eval.toSummaryString("\nResults\n******\n", true));
              System.out.println(eval.fMeasure(1) + " " + eval.precision(1)+ " " + eval.recall(1));
        
    }
}
