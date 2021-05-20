import sys
sys.path.insert(0, "/home/apprenant/Documents/Brief-Emotion-Analysis-Text/")

import streamlit as st # type: ignore
from src.app.models.regression_logistique import *

def app():
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.title('Prédictions et résultats des differents models utlisés')
    

    method = st.sidebar.radio('modeles', ['Régression logistique', 'XGBoost', 'Random Forest', 'SVM', 'Reseau de neurones'])
    
    
    if method == 'Régression logistique' : 

        st.markdown('Régression logistique')
        st.header('Etape 1')
        st.subheader('Database 1')
        y_pred, dataframe = predict_model(model, text_test, X_test)
        st.dataframe(dataframe)
        accuracy1 = calculate_score(model, text_test, y_test)
        st.write("Accuracy :", accuracy1)
        figure = plot_roc(y_test, y_pred)
        st.pyplot(figure)

        st.subheader('Database 2')
        y_pred2, dataframe2 = predict_model(model2, text_test2, X_test2)
        st.dataframe(dataframe2)
        accuracy2 = calculate_score(model2, text_test2, y_test2)
        st.write("Accuracy :", accuracy2)
        figure = plot_roc(y_test2, y_pred2)
        st.pyplot(figure)

        st.header('Etape 2')
        st.subheader('Databases combinées')
        y_pred3, dataframe3 = predict_model(model3, text_test3, X_test3)
        st.dataframe(dataframe3)
        accuracy3 = calculate_score(model3, text_test3, y_test3)
        st.write("Accuracy :", accuracy3)
        figure = plot_roc(y_test3, y_pred3)
        st.pyplot(figure)

        st.header('Etape 3')
        y_pred4, dataframe4 = predict_model(model4, text_test4, X_test4)
        st.dataframe(dataframe4)
        accuracy4 = calculate_accuracy_score(y_test4, y_pred4)
        st.write("Accuracy :", accuracy4)

    elif method == 'XGBoost' :
        st.write('XGBoost')
    elif method == 'Random Forest' :
        st.write('Random Forest')
    elif method == 'SVM' :
        st.write('SVM')
    else : 
        st.write('Reseau de neurones')
