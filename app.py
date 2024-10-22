from langchain_openai import ChatOpenAI
from langchain_experimental.agents import create_csv_agent
from dotenv import load_dotenv
import os
import streamlit as st
import pandas as pd

load_dotenv()

prompt = """Estas reglas las usarás para verificar la calidad de datos de las columnas del dataframe:
    Regla 1: El dato no puede ser nulo o vacío
    Regla 2: El dato debe no se puede repetir
    Regla 3: La longitud mínima de carácteres es de 3
    Regla 4: El dato debe ser alfanúmerico y solo admitir los caracteres especiales - o #
    Regla 5: El dato debe ser númerico
    Regla 6: La longitud mínima de carácteres es de 9 y máximo 11
    Regla 7: La longitud mínima de carácteres es de 8 y máximo 11
    Columnas y verificaciones:
    -NAME1: Aplicar Regla 1, Regla 2 y Regla 3
    -ORT01: Aplicar Regla 1
    -STRAS: Aplicar Regla 1 y Regla 4
    -STCD1: Aplicar Regla 1, Regla 2, Regla 5 y Regla 6
    -TELF1: Aplicar Regla 1, Regla 5 y Regla 7
    Formato de respuesta por cada columna(No incluyas más información):
    'Nombre de la columna': 'Regla que aplica: Cumple/No cumple'\n"""


def create_model():
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0,
        api_key=os.environ["OPENAI_API_KEY"],
        organization="org-nD7nJZT0fOaUsdRFdrrAa9Hu",
    )
    return llm


# Función para crear el agente
def create_agent(csv_file, llm_model):
    return create_csv_agent(
        llm_model,
        csv_file,
        pandas_kwargs={"delimiter": ";"},
        agent_type="openai-tools",
        verbose=True,
        allow_dangerous_code=True,
    )


def main():

    # Descripción de la página
    st.set_page_config(page_title="EY-DPB", page_icon="🤖")
    st.title("Data Profiler Bot🤖")
    st.write(
        """
        Esta PoC utiliza inteligencia artificial generativa para hacer perfilamiento de datos. 
        A continuación, se describen las reglas de verificación aplicadas sobre columnas 
        principales del archivo de testeo.
        """
    )

    # Datos de las reglas
    rules_data = {
        "Descripción de las reglas de validación": [
            "1. El dato no puede ser nulo o vacío",
            "2. El dato no debe repetirse",
            "3. La longitud mínima es de 3",
            "4. Alfanumérico y solo admitir caracteres especiales '- ó #'",
            "5. El dato debe ser numérico",
            "6. La longitud de caracteres mínima es de 9 y máxima de 11",
            "7. La longitud de caracteres mínima es de 8 y máxima de 11",
        ],
        "NAME1": [
            "Aplicar",
            "Aplicar",
            "Aplicar",
            "",
            "",
            "",
            "",
        ],
        "ORT01": [
            "Aplicar",
            "",
            "",
            "",
            "",
            "",
            "",
        ],
        "STRAS": [
            "Aplicar",
            "",
            "",
            "Aplicar",
            "",
            "",
            "",
        ],
        "STCD1": [
            "Aplicar",
            "Aplicar",
            "",
            "",
            "Aplicar",
            "Aplicar",
            "",
        ],
        "TELF1": [
            "Aplicar",
            "",
            "",
            "",
            "Aplicar",
            "",
            "Aplicar",
        ],
    }

    # Crear DataFrames
    rules_df = pd.DataFrame(rules_data)
    st.dataframe(rules_df, hide_index=True, use_container_width=True)

    # Lee el archivo CSV
    uploaded_file = st.file_uploader("Cargar archivo CSV", type=["csv"])
    # Verifica si fue subido el archivo
    if uploaded_file is not None:
        # Guarda el archivo en carpeta
        file_path = os.path.join("files/", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())  # Guardar el archivo cargado
        if st.button("Validar"):
            # Creación del modelo
            model = create_model()
            # Ejecución del modelo
            agent_executor = create_agent(file_path, model)
            # Respuesta del modelo
            response = agent_executor.invoke(prompt)
            # Muestra los resultados
            st.subheader("Resultados de la Verificación")
            st.write(response["output"])


if __name__ == "__main__":
    main()
