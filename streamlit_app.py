import streamlit as st
import eda, predict

with st.sidebar:
    st.title('Page Navigation')
    # input
    page = st.radio('Page', ('Eda', 'Model Demo'))

    st.write('# About')
    st.write('''Page ini adalah informasi data dan demi dari model prediksi player rating''')

if page == "EDA":
    eda.run()
else :
    predict.run()
