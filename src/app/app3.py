import streamlit as st
import pandas as pd
import joblib
import sys
sys.path.insert(0, "/home/apprenant/Documents/Brief-Emotion-Analysis-Text/")

def app():
    st.header('Comparaison des différents modèles')
    dataframe, accuracy1, dataframe2, accuracy2, dataframe3, accuracy3, dataframe4, accuracy4, y_proba, y_proba2, y_proba3, y_test, y_test2, y_test3 = joblib.load("regression_logistique.sav")
    dataframe5, accuracy5, dataframe6, accuracy6, dataframe7, accuracy7, dataframe8, accuracy8, y_pred5, y_pred6, y_pred7, y_proba5, y_proba6, y_proba7, y_test5, y_test6, y_test7 = joblib.load("xg_boost.sav")
    dataframe9, dataframe10, dataframe11, dataframe12, train_score1, train_score2, train_score3, train_score4, train_score5, train_score6, accuracy9 = joblib.load("random_forest.sav")
    dataframe13, accuracy10, dataframe14, accuracy11, dataframe15, accuracy12, dataframe16, accuracy13 = joblib.load("svm_model.sav")
    score, score2, score3, score4 = joblib.load("neural_network.sav")
    results_table =pd.DataFrame({'Régression logistique (accuracy)' : [accuracy1, accuracy2, accuracy3, accuracy4], 
                                'XGBoost (accuracy)':[accuracy5, accuracy6, accuracy7, accuracy8],
                                'Random Forest (train score - training set, train score - testing set)': [[train_score1,train_score2],[train_score3,train_score4],[train_score5,train_score6], accuracy9],
                                'SVM (accuracy)': [accuracy10, accuracy11, accuracy12, accuracy13],
                                'Neural Network (test loss, test accuracy)': [[round(score[0], 5), round(score[1], 5)],[round(score2[0], 5), round(score2[1], 5)],[round(score3[0], 5), round(score3[1], 5)], [round(score4[0], 5), round(score4[1], 5)]]},
                                 index = ['Etape 1 (Dataframe 1)', 'Etape 1 (Dataframe 2)', 'Etape 2', 'Etape 3'])
    st.dataframe(results_table)