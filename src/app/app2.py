import sys
from numpy.lib.npyio import load
from streamlit.proto.DataFrame_pb2 import DataFrame
sys.path.insert(0, "/home/apprenant/Documents/Brief-Emotion-Analysis-Text/")

import streamlit as st
import joblib
from src.functions import *



def app():
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.title('Prédictions et résultats des differents models utlisés')
    
    # Define a selection of the classifiers
    method = st.sidebar.radio('Modèles', ['Régression logistique', 'XGBoost', 'Random Forest', 'SVM', 'Reseau de neurones'])
    
    # Condition to display the results of each classifier
    if method == 'Régression logistique' : 
        st.markdown('Régression logistique')
        dataframe, accuracy1, dataframe2, accuracy2, dataframe3, accuracy3, dataframe4, accuracy4, y_proba, y_proba2, y_proba3, y_test, y_test2, y_test3 = joblib.load("regression_logistique.sav")
        st.header('Etape 1')
        st.subheader('Database 1')
        st.dataframe(dataframe)
        st.write("Accuracy :", accuracy1)
        figure = plot_roc(y_test, y_proba)
        st.pyplot(figure)

        st.subheader('Database 2')
        st.dataframe(dataframe2)
        st.write("Accuracy :", accuracy2)
        figure = plot_roc(y_test2, y_proba2)
        st.pyplot(figure)

        st.header('Etape 2')
        st.subheader('Databases combinées')
        st.dataframe(dataframe3)
        st.write("Accuracy :", accuracy3)
        figure = plot_roc(y_test3, y_proba3)
        st.pyplot(figure)

        st.header('Etape 3')
        st.dataframe(dataframe4)
        st.write("Accuracy :", accuracy4)

    elif method == 'XGBoost' :
        st.markdown('XGBoost')
        dataframe5, accuracy5, dataframe6, accuracy6, dataframe7, accuracy7, dataframe8, accuracy8, y_pred5, y_pred6, y_pred7, y_proba5, y_proba6, y_proba7, y_test5, y_test6, y_test7 = joblib.load("xg_boost.sav")
        st.header('Etape 1')
        st.subheader('Database 1')
        st.dataframe(dataframe5)
        st.write("Accuracy :", accuracy5)
        figure = plot_roc(y_test5, y_proba5)
        st.pyplot(figure)

        st.subheader('Database 2')
        st.dataframe(dataframe6)
        st.write("Accuracy :", accuracy6)
        figure = plot_roc(y_test6, y_proba6)
        st.pyplot(figure)

        st.header('Etape 2')
        st.subheader('Databases combinées')
        st.dataframe(dataframe7)
        st.write("Accuracy :", accuracy7)
        figure = plot_roc(y_test7, y_proba7)
        st.pyplot(figure)

        st.header('Etape 3')
        st.dataframe(dataframe8)
        st.write("Accuracy :", accuracy8)

    elif method == 'Random Forest' :
        st.markdown('Random Forest')
        dataframe9, dataframe10, dataframe11, dataframe12, train_score1, train_score2, train_score3, train_score4, train_score5, train_score6, accuracy9 = joblib.load("random_forest.sav")
        st.header('Etape 1')
        st.subheader('Database 1')
        st.dataframe(dataframe9)
        st.write("Train score (train) :", train_score1)
        st.write("Train score (test) :", train_score2)
        

        st.subheader('Database 2')
        st.dataframe(dataframe10)
        st.write("Train Score (train) :", train_score3)
        st.write("Train score (test) :", train_score4)
        

        st.header('Etape 2')
        st.subheader('Databases combinées')
        st.dataframe(dataframe11)
        st.write("Train Score (train) :", train_score5)
        st.write("Train score (test) :", train_score6)
        

        st.header('Etape 3')
        st.dataframe(dataframe12)
        st.write("Accuracy :", accuracy9)
        
    elif method == 'SVM' :
        st.markdown('SVM')
        dataframe13, accuracy10, dataframe14, accuracy11, dataframe15, accuracy12, dataframe16, accuracy13 = joblib.load("svm_model.sav")
        st.header('Etape 1')
        st.subheader('Database 1')
        st.dataframe(dataframe13)
        st.write("Accuracy :", accuracy10)
        

        st.subheader('Database 2')
        st.dataframe(dataframe14)
        st.write("Accuracy :", accuracy11)
        

        st.header('Etape 2')
        st.subheader('Databases combinées')
        st.dataframe(dataframe15)
        st.write("Accuracy :", accuracy12)
        

        st.header('Etape 3')
        st.dataframe(dataframe16)
        st.write("Accuracy :", accuracy13)
    else : 
        st.markdown('Reseau de neurones')
        score, score2, score3, score4 = joblib.load("neural_network.sav")
        st.header('Etape 1')
        st.subheader('Database 1')
        st.write('Test loss:', round(score[0], 5))
        st.write('Test accuracy:', round(score[1], 5))

        st.subheader('Database 2')
        st.write('Test loss:', round(score2[0], 5))
        st.write('Test accuracy:', round(score2[1], 5))

        st.header('Etape 2')
        st.write('Test loss:', round(score3[0], 5))
        st.write('Test accuracy:', round(score3[1], 5))

        st.header('Etape 3')
        st.write('Test loss:', round(score4[0], 5))
        st.write('Test accuracy:', round(score4[1], 5))

