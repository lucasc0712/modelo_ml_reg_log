import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Configuração da página
st.set_page_config(
    page_title="Predição de Cancelamento",
    page_icon="📊",
    layout="centered"
)

# Título e Descrição
st.title("📊 Predição de Cancelamento de Clientes")
st.write("""
Esta aplicação utiliza um modelo de **Regressão Logística** para prever a probabilidade de cancelamento (Churn) de um cliente com base em seu comportamento de uso.
""")

st.markdown("---")

st.write("""
A precisão deste modelo é probabilística e pode expressar falsos posivitos ou falsos negativos. Os dados exibidos devem ser tratados apenas como um indicativo.
""")

st.markdown("---")

# Função para carregar o modelo e o scaler salvos via pickle
@st.cache_resource
def carregar_modelo_e_scaler():
    modelo = pickle.load(open('classificador_logistico.sav', 'rb'))
    scaler = pickle.load(open('scaler_logistico.sav', 'rb'))
    return modelo, scaler

try:
    modelo, scaler = carregar_modelo_e_scaler()
except FileNotFoundError as e:
    st.error("Erro: Certifique-se de que os arquivos 'classificador_logistico.sav' e 'scaler_logistico.sav' estão no mesmo diretório da aplicação.")
    st.stop()

# Formulário para entradas do usuário
st.subheader("📥 Insira os dados do Cliente")

with st.form("form_predicao"):
    col1, col2 = st.columns(2)
    
    with col1:
        total_day_minutes = st.number_input(
            "Total de Minutos Diurnos",
            min_value=0.0,
            max_value=1000.0,
            value=180.0,
            step=1.0,
            help="Total de minutos de atendimento consumidos durante o período diurno."
        )
        
    with col2:
        customer_service_calls = st.number_input(
            "Chamadas ao Suporte",
            min_value=0,
            max_value=20,
            value=1,
            step=1,
            help="Número de vezes que o cliente entrou em contato com o suporte."
        )
        
    btn_submeter = st.form_submit_button("Realizar Predição", type="primary", use_container_width=True)

# Processamento e exibição do resultado
if btn_submeter:
    # 1. Monta o DataFrame com as mesmas colunas do treinamento
    dados_entrada = pd.DataFrame({
        "Total_day_minutes": [float(total_day_minutes)],
        "Customer_service_calls": [float(customer_service_calls)]
    })

    # 2. Aplica a padronização utilizando o scaler carregado
    dados_escalonados = scaler.transform(dados_entrada)

    # 3. Predição do modelo
    predicao = modelo.predict(dados_escalonados)[0]
    probabilidade = modelo.predict_proba(dados_escalonados)[0]

    prob_permanencia = probabilidade[0] * 100
    prob_churn = probabilidade[1] * 100

    st.markdown("---")
    st.subheader("🎯 Resultado da Análise")

    col_res1, col_res2 = st.columns(2)

    with col_res1:
        st.metric(label="Chance de Permanência", value=f"{prob_permanencia:.1f}%")

    with col_res2:
        st.metric(label="Chance de Cancelamento", value=f"{prob_churn:.1f}%")

    # Exibição do diagnóstico visual
    if predicao == 1:
        st.error(f"⚠️ **Alto Risco de Cancelamento!** O cliente possui **{prob_churn:.2f}%** de chance de cancelar os serviços.")
    else:
        st.success(f"✅ **Baixo Risco de Cancelamento.** O cliente possui apenas **{prob_churn:.2f}%** de chance de cancelar os serviços.")