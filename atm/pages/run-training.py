import streamlit as st

from model.pipelines.train_pipeline import train_pipeline 
from model.components.config import ModelConfig

split_cols = True
test_set_options = {'Split from dataset': not split_cols, 'Use seperate test set':split_cols}



def train():
    if st.session_state.seperate_test_set:
        print("inside sep data set")
        if st.session_state.first_path and st.session_state.second_path:
            train_pipeline(
                [st.session_state.first_path, st.session_state.second_path],
                st.session_state.ml_alogrithm.strip(),
                st.session_state.random_state,
                use_ui=True,
                seperate_test_set=True
            )
        else:
            return
    else:
        print(f"inside single data: {st.session_state.first_path}")
        if st.session_state.first_path:
            print('consisted first path')
            train_pipeline(
                [st.session_state.first_path],
                st.session_state.ml_alogrithm.strip(),
                st.session_state.random_state,
                use_ui=True,
                seperate_test_set=False
            )
        else:
            return


if 'training_status' not in st.session_state:
    st.session_state.training_status = False

if 'seperate_test_set' not in st.session_state:
    st.session_state.seperate_test_set = False

if 'first_path' not in st.session_state:
    st.session_state.first_path = ''

if 'second_path' not in st.session_state:
    st.session_state.second_path = ''

if 'random_state' not in st.session_state:
    st.session_state.random_state = 0

if 'ml_alogrithm' not in st.session_state:
    st.session_state.ml_alogrithm = ''



col_radio, col_input  = st.columns([0.3, 0.7])

with col_radio:
    st.write("Choose test set option")
    selected_option = st.radio("Select the appropriate", test_set_options.keys()).strip()

with col_input:
    st.write("Enter data path/s")

    if test_set_options[selected_option]:
        st.session_state.seperate_test_set = True

        col_first, col_second = st.columns(2)

        with col_first:
            st.session_state.first_path = st.text_input('Enter data path').strip()

        with col_second:
            st.session_state.second_path = st.text_input('Enter data path for seperate test data.').strip()
    else:
        st.session_state.first_path = st.text_input('Enter data path').strip()
        st.session_state.seperate_test_set = False


st.write('---')

st.session_state.random_state = st.slider("Choose Random State", 0, 100)

st.write('---')

st.session_state.ml_alogrithm = st.radio('Select target algorithm', ModelConfig.model_names)

st.write('---')

st.write("Click train to start training the model")
st.button("Train", on_click=train)
