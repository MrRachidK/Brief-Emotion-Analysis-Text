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
    

    method = st.sidebar.radio('Modèles', ['Régression logistique', 'XGBoost', 'Random Forest', 'SVM', 'Reseau de neurones'])
    
    
    if method == 'Régression logistique' : 
        st.markdown('Régression logistique')
        dataframe, accuracy1, dataframe2, accuracy2, dataframe3, accuracy3, dataframe4, accuracy4, y_pred, y_pred2, y_pred3, y_test, y_test2, y_test3 = joblib.load("regression_logistique.sav")
        st.header('Etape 1')
        st.subheader('Database 1')
        st.dataframe(dataframe)
        st.write("Accuracy :", accuracy1)
        figure = plot_roc(y_test, y_pred)
        st.pyplot(figure)

        st.subheader('Database 2')
        st.dataframe(dataframe2)
        st.write("Accuracy :", accuracy2)
        figure = plot_roc(y_test2, y_pred2)
        st.pyplot(figure)

        st.header('Etape 2')
        st.subheader('Databases combinées')
        st.dataframe(dataframe3)
        st.write("Accuracy :", accuracy3)
        figure = plot_roc(y_test3, y_pred3)
        st.pyplot(figure)

        st.header('Etape 3')
        st.dataframe(dataframe4)
        st.write("Accuracy :", accuracy4)

    elif method == 'XGBoost' :
        st.markdown('XGBoost')
        dataframe5, accuracy5, dataframe6, accuracy6, dataframe7, accuracy7, dataframe8, accuracy8, y_pred5, y_pred6, y_pred7, y_test5, y_test6, y_test7 = joblib.load("xg_boost.sav")
        st.header('Etape 1')
        st.subheader('Database 1')
        st.dataframe(dataframe5)
        st.write("Accuracy :", accuracy5)
        figure = plot_roc(y_test5, y_pred5)
        st.pyplot(figure)

        st.subheader('Database 2')
        st.dataframe(dataframe6)
        st.write("Accuracy :", accuracy6)
        figure = plot_roc(y_test6, y_pred6)
        st.pyplot(figure)

        st.header('Etape 2')
        st.subheader('Databases combinées')
        st.dataframe(dataframe7)
        st.write("Accuracy :", accuracy7)
        figure = plot_roc(y_test7, y_pred7)
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
        # cross_val_score1 = calculate_cross_val_score(loaded_model9, text_train9, y_train9, 5)
        # st.write("Cross Validation Score :", cross_val_score1)
        st.write("Train score (train) :", train_score1)
        st.write("Train score (test) :", train_score2)
        

        st.subheader('Database 2')
        st.dataframe(dataframe10)
        # cross_val_score2 = calculate_cross_val_score(loaded_model10, text_train10, y_train10, 5)
        # st.write("Cross Validation Score :", cross_val_score2)
        st.write("Train Score (train) :", train_score3)
        st.write("Train score (test) :", train_score4)
        

        st.header('Etape 2')
        st.subheader('Databases combinées')
        st.dataframe(dataframe11)
        # cross_val_score3 = calculate_cross_val_score(loaded_model11, text_train11, y_train11, 5)
        # st.write("Cross Validation Score :", cross_val_score3)
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
        st.write('Reseau de neurones')
