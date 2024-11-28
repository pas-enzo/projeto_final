import tensorflow as tf
import numpy as np
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing import image
import streamlit as st
from PIL import Image

# Função para carregar o modelo 
@st.cache_resource
def carregar_modelo(json_path, weights_path):
    try:
        # Carregar a arquitetura
        with open(json_path, 'r') as file:
            model_json = file.read()
        model = model_from_json(model_json)
        
        # Carregar os pesos 
        model.load_weights(weights_path)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model
    except Exception as e:
        raise ValueError(f"Erro ao carregar o modelo: {e}")

# Função para processar e classificar uma imagem
def classificar_imagem(modelo, imagem):
    try:
        # Redimensionar e normalizar a imagem
        img = imagem.resize((64, 64))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Realizar a predição
        predicao = modelo.predict(img_array)
        classe = np.argmax(predicao, axis=1)

        # Interpretar a classe
        if classe[0] == 0:
            return "Gato"
        else:
            return "Cachorro"
    except Exception as e:
        return f"Erro ao classificar a imagem: {e}"

# Interface do Streamlit
st.title("Classificador de Imagens: Gato ou Cachorro?")
st.write("Faça upload de uma imagem para classificá-la como Gato ou Cachorro.")

# Upload do modelo
json_path = st.text_input("Caminho para o arquivo JSON do modelo:", "network.json")
weights_path = st.text_input("Caminho para os pesos do modelo (arquivo .h5):", "weights.hdf5")

# Carregar o modelo
if json_path and weights_path:
    try:
        modelo = carregar_modelo(json_path, weights_path)
        st.success("Modelo carregado com sucesso!")
    except Exception as e:
        st.error(f"Erro ao carregar o modelo: {e}")

# Upload da imagem
uploaded_file = st.file_uploader("Faça upload de uma imagem", type=["jpg", "jpeg", "png"])

if uploaded_file and 'modelo' in locals():
    try:
        # Abrir e mostrar a imagem carregada
        img = Image.open(uploaded_file)
        st.image(img, caption="Imagem carregada", use_column_width=True)

        # Classificar a imagem
        resultado = classificar_imagem(modelo, img)
        st.success(f"A imagem carregada é um: **{resultado}**")
    except Exception as e:
        st.error(f"Erro ao processar a imagem: {e}")
