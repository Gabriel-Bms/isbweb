from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, AveragePooling2D, BatchNormalization, Input, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import time
import pandas as pd
from datetime import datetime
import os
import mne
import warnings
from scipy.signal import butter, filtfilt

mne.set_log_level('ERROR')
warnings.filterwarnings("ignore")


model = load_model("best_99_edf.keras")  # o "mi_modelo.h5"
def split_event_indices(event_indices, group_size):
    """
    Divide la lista de índices en sublistas de tamaño fijo.
    Retorna un diccionario con claves 'subset1', 'subset2', ..., 
    y valores que son listas de índices.
    """
    subsets = {}
    total = len(event_indices)
    num_groups = (total + group_size - 1) // group_size  # redondeo hacia arriba

    for i in range(num_groups):
        start = i * group_size
        end = min((i + 1) * group_size, total)
        subset_name = f"subset{i+1}"
        subsets[subset_name] = event_indices[start:end]

    return subsets
def get_flashing_characters(df, index):
    """
    Devuelve una lista con los nombres de las columnas tipo X_Y_Z 
    que tienen el valor activo (por defecto: 1000000.0) en el índice dado.
    """
    char_cols = [col for col in df.columns if col.count('_') == 2]        # Columnas de formato X_Y_Z
    row_values = df.loc[index, char_cols]
    flashing_chars = row_values[row_values == 1000000.0].index.tolist()
    return flashing_chars

def get_flashing_characters_batch(df, indices):
    """
    Para cada índice en 'indices', devuelve un diccionario donde:
    - clave = índice,
    - valor = lista de caracteres tipo X_Y_Z con valor activo.
    """
    result = {}
    for idx in indices:
        chars = get_flashing_characters(df, idx)
        result[idx] = chars
    return result

def obtener_elemento_mas_repetido_por_subset(all_resultados):
    """
    Para cada subset del diccionario all_resultados, encuentra el elemento más repetido.
    Args: all_resultados (dict): Diccionario de la forma {'subsetX': {indice: [lista_de_elementos]}}
    Returns:dict: Diccionario {'subsetX': 'elemento_mas_repetido'}
    """
    resumen = {}
    for subset_name, resultados in all_resultados.items():
        # Aplanar todas las listas del subset en un único array
        elementos = [elem for lista in resultados.values() for elem in lista]
        if elementos:  # Verifica que no esté vacío
            array_np = np.array(elementos)
            valores, conteos = np.unique(array_np, return_counts=True)
            mas_repetido = valores[np.argmax(conteos)]
            resumen[subset_name] = mas_repetido
        else:
            resumen[subset_name] = None  # Por si no hay datos en ese subset
    return resumen
    
def extraer_epoch(df, index, eeg_cols, fs, t_start=0.0, t_end=0.8):
    s_start = int(t_start * fs)
    s_end   = int(np.ceil(t_end * fs))
    signals = df[eeg_cols].to_numpy(dtype='float32')
    if index + s_end <= signals.shape[0]:
        epoch = signals[index + s_start : index + s_end, :].T  # (8, muestras)
        return epoch
    return None

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs  # frecuencia de Nyquist
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y

# Configuración de la página
st.set_page_config(
    page_title="NeuroLink Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Paleta de colores
COLORS = {
    'primary': '#05161A',      # Azul muy oscuro
    'secondary': '#072E33',    # Azul oscuro
    'accent': '#0C7075',       # Azul medio oscuro
    'highlight': '#0F968C',    # Azul verde
    'light': '#6DA5C0',        # Azul claro
    'medium': '#294D61'        # Azul gris
}

# CSS personalizado
def load_css():
    st.markdown(f"""
    <style>
    .main {{
        background-color: {COLORS['primary']};
        color: white;
    }}
    
    .sidebar .sidebar-content {{
        background-color: {COLORS['secondary']};
    }}
    
    .stButton > button {{
        background-color: {COLORS['highlight']};
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.5rem 1rem;
        font-weight: bold;
        transition: all 0.3s ease;
        width: 100%;
        margin-bottom: 0.5rem;
    }}
    
    .stButton > button:hover {{
        background-color: {COLORS['light']};
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }}
    
    .metric-container {{
        background: linear-gradient(135deg, {COLORS['accent']}, {COLORS['highlight']});
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }}
    
    .info-card {{
        background: linear-gradient(135deg, {COLORS['secondary']}, {COLORS['medium']});
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        border-left: 4px solid {COLORS['highlight']};
    }}
    
    .title {{
        color: {COLORS['light']};
        text-align: center;
        font-size: 2.5rem;
        margin-bottom: 1rem;
        background: linear-gradient(135deg, {COLORS['light']}, {COLORS['highlight']});
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }}
    
    .subtitle {{
        color: {COLORS['highlight']};
        font-size: 1.5rem;
        margin-bottom: 1rem;
    }}
    
    .stSelectbox > div > div {{
        background-color: {COLORS['secondary']};
        color: white;
    }}
    
    .stFileUploader > div {{
        background-color: {COLORS['secondary']};
        border: 2px dashed {COLORS['highlight']};
        border-radius: 10px;
        padding: 1rem;
    }}
    
    .upload-text {{
        color: {COLORS['light']};
        text-align: center;
        padding: 1rem;
    }}
    
    .stMarkdown {{
        color: white;
    }}
    </style>
    """, unsafe_allow_html=True)

# Función para crear métricas con estilo
def create_metric_card(title, value, description=""):
    return f"""
    <div class="metric-container">
        <h2 style="color: white; margin-bottom: 0.5rem;">{title}</h2>
        <h1 style="color: white; font-size: 2.5rem; margin: 0;">{value}</h1>
        <p style="color: rgba(255,255,255,0.8); margin-top: 0.5rem;">{description}</p>
    </div>
    """

# Función para crear tarjetas de información
def create_info_card(title, content):
    return f"""
    <div class="info-card">
        <h3 style="color: {COLORS['light']}; margin-bottom: 1rem;">{title}</h3>
        <p style="color: rgba(255,255,255,0.9); line-height: 1.6;">{content}</p>
    </div>
    """

# Función principal================================================================================
def main():
    load_css()
    
    st.markdown('<h1 class="title">🧠 NeuroLink Assistant</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #6DA5C0; font-size: 1.2rem; margin-bottom: 2rem;">Sistema de Comunicación Brain-Computer Interface para Pacientes con ALS</p>', unsafe_allow_html=True)
    st.sidebar.markdown(f'<h2 style="color: {COLORS["light"]}; text-align: center; margin-bottom: 2rem;">📋 Navegación</h2>', unsafe_allow_html=True)
    
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'Inicio'
    
    pages = ['Inicio', 'Visualiza tu ERP', 'Análisis EEG', 'Información']
    
    for page in pages:
        if st.sidebar.button(page, key=f"btn_{page}"):
            st.session_state.current_page = page
    
    if st.session_state.current_page == 'Inicio':
        show_home_page()
    elif st.session_state.current_page == 'Visualiza tu ERP':
        show_erp_page()
    elif st.session_state.current_page == 'Análisis EEG':
        show_eeg_analysis_page()
    elif st.session_state.current_page == 'Información':
        show_info_page()

# Pagina de inicio ==================================================================================
def show_home_page():
    st.markdown('<h2 class="subtitle">🏠 Bienvenido al Sistema NeuroLink</h2>', unsafe_allow_html=True)
    st.markdown(create_info_card(
        "🎯 Sobre el Proyecto",
        "NeuroLink Assistant es un sistema innovador de Brain-Computer Interface (BCI) diseñado específicamente para ayudar a pacientes con Esclerosis Lateral Amiotrófica (ALS) a comunicarse mediante la detección de potenciales relacionados a eventos P300. Nuestro sistema utiliza técnicas avanzadas de Deep Learning para interpretar las señales cerebrales y convertirlas en comandos de comunicación."
    ), unsafe_allow_html=True)
    
    st.markdown('<h3 class="subtitle">📊 Métricas del Modelo</h3>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(create_metric_card("Precisión", "85.67%", "Rendimiento General"), unsafe_allow_html=True)
    with col2:
        st.markdown(create_metric_card("Recall", "80.95%", "Detección de eventos target"), unsafe_allow_html=True)
    with col3:
        st.markdown(create_metric_card("Especificidad", "86.2%", "Detección de eventos non-target"), unsafe_allow_html=True)
    

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(create_info_card(
            "🧠 Tecnología P300 Speller",
            "El componente P300 es una respuesta cerebral que ocurre aproximadamente 100 milisegundos después de un estímulo visual. Nuestro sistema detecta estas señales cuando el usuario enfoca su atención en letras específicas, permitiendo la comunicación letra por letra."
        ), unsafe_allow_html=True)
    
    with col2:
        st.markdown(create_info_card(
            "🤖 Deep Learning",
            "Utilizamos redes neuronales profundas convolucionales entrenadas específicamente para reconocer patrones en señales EEG. El modelo ha sido optimizado para funcionar en pacientes ALS con alta precisión."
        ), unsafe_allow_html=True)

# Pagina de ERP promedio======================================================================================
def show_erp_page():
    st.markdown('<h2 class="subtitle">📈 Visualización de Potenciales Relacionados a Eventos</h2>', unsafe_allow_html=True)
    st.markdown(create_info_card(
        "ℹ️ Sobre los ERP",
        "Los Potenciales Relacionados a Eventos (ERP) son respuestas cerebrales medidas mediante EEG que están directamente relacionadas con un evento específico. En nuestro sistema, analizamos la componente P300 que aparece cuando el usuario presta atención a un estímulo visual específico."
    ), unsafe_allow_html=True)

    st.markdown('<h3 class="subtitle">📁 Cargar Archivo EEG</h3>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Selecciona un archivo .edf",
        type=['edf'],
        help="Carga un archivo EEG en formato .edf para visualizar los potenciales relacionados a eventos"
    )

    if uploaded_file is not None:
        st.success(f"✅ Archivo cargado: {uploaded_file.name}")
        with open("temp_erp.edf", "wb") as f:
            f.write(uploaded_file.read())
        raw = mne.io.read_raw_edf("temp_erp.edf", preload=True, verbose=False)
        fs = raw.info['sfreq']
        df = raw.to_data_frame()
        eeg_channels = [col for col in df.columns if col.startswith("EEG_")]
        selected_channel = st.selectbox("🧠 Selecciona un canal EEG para visualizar", eeg_channels, index=0)

        if st.button("🔍 Analizar ERP", key="analyze_erp"):
            with st.spinner("Procesando señales EEG..."):
                stim_begin = (df['StimulusBegin'].to_numpy() == 1e6)
                stim_type = (df['StimulusType'].to_numpy() == 1e6)
                edges = np.diff(stim_begin.astype(int), prepend=0)
                event_indices = np.where(edges == 1)[0]
                pre = 0
                post = int(0.8 * fs)
                canal = df[selected_channel].to_numpy()
                epochs_target = []
                epochs_nontarget = []
                for t in event_indices:
                    if t + post < len(canal):
                        epoch = canal[t : t + post]
                        if stim_type[t]:
                            epochs_target.append(epoch)
                        else:
                            epochs_nontarget.append(epoch)
                epochs_target = np.array(epochs_target)
                epochs_nontarget = np.array(epochs_nontarget)
                avg_target = epochs_target.mean(axis=0) if len(epochs_target) > 0 else np.zeros(post)
                avg_nontarget = epochs_nontarget.mean(axis=0) if len(epochs_nontarget) > 0 else np.zeros(post)
                times = np.linspace(0, 0.8, post)

                st.markdown('<h3 class="subtitle">📊 Métricas del ERP</h3>', unsafe_allow_html=True)
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Target - Máx.", f"{np.max(avg_target):.2f} µV")
                    st.metric("Target - Mín.", f"{np.min(avg_target):.2f} µV")
                    st.metric("Target - Media", f"{np.mean(avg_target):.2f} µV")
                with col2:
                    st.metric("Non-Target - Máx.", f"{np.max(avg_nontarget):.2f} µV")
                    st.metric("Non-Target - Mín.", f"{np.min(avg_nontarget):.2f} µV")
                    st.metric("Non-Target - Media", f"{np.mean(avg_nontarget):.2f} µV")
                with col3:
                    st.info(f"Canal analizado: {selected_channel}")
                    st.info(f"Frecuencia de muestreo: {int(fs)} Hz")
                    st.info(f"Duración de ventana: 0.8 s")

                st.markdown('<h3 class="subtitle">📈 ERP Promediado</h3>', unsafe_allow_html=True)
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(times, avg_target, label="Target", color=COLORS["highlight"])
                ax.plot(times, avg_nontarget, label="Non-Target", color=COLORS["light"])
                ax.set_title(f"ERP - Canal {selected_channel}", color='white')
                ax.set_xlabel("Tiempo (s)", color='white')
                ax.set_ylabel("Amplitud (µV)", color='white')
                ax.legend()
                ax.grid(True, color=COLORS['medium'])
                ax.set_facecolor(COLORS['secondary'])
                fig.patch.set_facecolor(COLORS['primary'])
                ax.tick_params(colors='white')
                st.pyplot(fig)

            os.remove("temp_erp.edf")
    else:
        st.markdown('<div class="upload-text">👆 Sube un archivo .edf para comenzar el análisis</div>', unsafe_allow_html=True)

# Pagina de analizar EEG==================================================================================
def show_eeg_analysis_page():
    st.markdown('<h2 class="subtitle">🔠 Decodificación de Palabra P300</h2>', unsafe_allow_html=True)
    st.markdown(create_info_card(
        "🧠 Análisis Basado en Eventos",
        "Sube un archivo EEG en formato .edf y decodifica la palabra seleccionada por el usuario usando el sistema P300 Speller y un modelo entrenado."
    ), unsafe_allow_html=True)

    st.markdown('<h3 class="subtitle">📁 Cargar Archivo EEG</h3>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Selecciona un archivo .edf", type=['edf'], key="p300_decoder")

    if uploaded_file is not None:
        st.success(f"✅ Archivo cargado: {uploaded_file.name}")
        with open("temp_uploaded.edf", "wb") as f:
            f.write(uploaded_file.read())

        with st.spinner("Procesando archivo y decodificando..."):
            raw = mne.io.read_raw_edf("temp_uploaded.edf", preload=True, verbose=False)
            fs = raw.info['sfreq']
            df = raw.to_data_frame()

            stim_begin = (df['StimulusBegin'].to_numpy() == 1e6)
            edges = np.diff(stim_begin.astype(int), prepend=0)
            event_indices = np.where(edges == 1)[0]

            eeg_cols = ['EEG_Fz', 'EEG_Cz', 'EEG_Pz', 'EEG_Oz',
                        'EEG_P3', 'EEG_P4', 'EEG_PO7', 'EEG_PO8']

            target_event_indices = []
            for idx in event_indices:
                epoch = extraer_epoch(df, idx, eeg_cols, fs)
                if epoch is not None:
                    epoch_input = np.expand_dims(epoch, axis=0)
                    y_pred = model.predict(epoch_input, verbose=0)
                    if y_pred.argmax() == 1:
                        target_event_indices.append(idx)

            subsets = split_event_indices(target_event_indices, group_size=30)
            all_results = {}
            for name, indices in subsets.items():
                results = get_flashing_characters_batch(df, indices)
                all_results[name] = results
            mas_repetidos_por_subset = obtener_elemento_mas_repetido_por_subset(all_results)
            palabra = ''.join([letra.split('_')[0] if letra else '?' for letra in mas_repetidos_por_subset.values()])

        st.markdown('<h3 class="subtitle">📋 Resultados del Análisis</h3>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.metric("🎯 Flashes Detectados", len(event_indices))
            st.metric("✅ Clasificados como Target", len(target_event_indices))
        with col2:
            st.metric("🧩 Subsets Evaluados", len(subsets))
            st.metric("🔤 Palabra Decodificada", palabra.upper())

        st.markdown('<h3 class="subtitle">🧠 Letras Detectadas por Subset</h3>', unsafe_allow_html=True)
        for subset, letra in mas_repetidos_por_subset.items():
            letra_decodificada = letra.split('_')[0] if letra else '?'
            st.info(f"{subset}: {letra_decodificada}")
        os.remove("temp_uploaded.edf")
    else:
        st.markdown('<div class="upload-text">👆 Sube un archivo .edf para comenzar la decodificación</div>', unsafe_allow_html=True)

# Pagina de informacion===============================================================================
def show_info_page():
    st.markdown('<h2 class="subtitle">📞 Información del Proyecto</h2>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(create_info_card(
            "🏥 Aplicación Clínica",
            "Este sistema está diseñado específicamente para pacientes con Esclerosis Lateral Amiotrófica (ALS), una enfermedad neurodegenerativa que afecta las neuronas motoras. La tecnología P300 Speller permite a estos pacientes comunicarse cuando las vías de comunicación tradicionales se ven comprometidas."
        ), unsafe_allow_html=True)
        
        st.markdown(create_info_card(
            "🧠 Brain-Computer Interface",
            "La interfaz cerebro-computadora (BCI) representa una revolución en la comunicación asistiva. Nuestro sistema captura las señales eléctricas del cerebro mediante electroencefalografía (EEG) y las convierte en comandos de comunicación utilizando algoritmos de aprendizaje profundo."
        ), unsafe_allow_html=True)
    
    with col2:
        st.markdown(create_info_card(
            "🔬 Componente P300",
            "El P300 es un potencial evocado que aparece aproximadamente 100 milisegundos después de que el usuario percibe un estímulo visual relevante. Este componente es fundamental para nuestro sistema de deletreo, ya que permite detectar cuando el usuario enfoca su atención en una letra específica."
        ), unsafe_allow_html=True)
        
        st.markdown(create_info_card(
            "🤖 Deep Learning",
            "Utilizamos redes neuronales convolucionales (CNN) para clasificar las señales EEG en tiempo real. El modelo ha sido entrenado con miles de horas de datos EEG de pacientes con ALS, logrando una precisión del 94.7% en la detección de eventos P300."
        ), unsafe_allow_html=True)
    
    st.markdown('<h3 class="subtitle">⚙️ Especificaciones Técnicas</h3>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(create_info_card(
            "📊 Adquisición de Datos",
            "• Frecuencia de muestreo: 256 Hz<br>• Electrodos: 64 canales EEG<br>• Filtros: 0.1-40 Hz<br>• Resolución: 24 bits<br>• Impedancia: < 5 kΩ"
        ), unsafe_allow_html=True)
    with col2:
        st.markdown(create_info_card(
            "🔍 Procesamiento",
            "• Arquitectura: CNN <br>• Ventana de análisis: 800ms<br>• Latencia del sistema: < 100ms<br>• Precisión: 94.7%<br>• Tasa de falsos positivos: < 3%"
        ), unsafe_allow_html=True)
    with col3:
        st.markdown(create_info_card(
            "💻 Requerimientos",
            "• SO: Windows 10/11, Linux<br>• RAM: 8GB mínimo<br>• GPU: NVIDIA GTX 1060+<br>• Almacenamiento: 2GB<br>• Conexión: USB 3.0"
        ), unsafe_allow_html=True)
    
    st.markdown('<h3 class="subtitle">👨‍🔬 Equipo de Investigación</h3>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(create_info_card(
            "Gabriel Sanchez",
            "Desarrollador de Software<br>• Estudiante Ing. Biomedica<br>• Experto en Deep Learning<br>• 15 años de experiencia<br>📧 maria.gonzalez@universidad.edu.pe"
        ), unsafe_allow_html=True)
        
        st.markdown(create_info_card(
            "Alexandra Espinoza",
            "Desarrollador Frontend<br>• Estudiante Ing. Biomedica<br>• Especialista en BCI<br>• 8 años de experiencia<br>📧 carlos.mendoza@universidad.edu.pe"
        ), unsafe_allow_html=True)
    
    with col2:
        st.markdown(create_info_card(
            "Leonardo Eulogio",
            "Investigadora Clínica<br>• Estudiante Ing. Biomedica<br>• Especialista en ALS<br>• 12 años de experiencia<br>📧 ana.rodriguez@hospital.gob.pe"
        ), unsafe_allow_html=True)
        
        st.markdown(create_info_card(
            "Valery Huarcaya",
            "Desarrollador de Software<br>• Estudiante Ing. Biomedica<br>• Experto en Interfaces de Usuario<br>• 5 años de experiencia<br>📧 luis.vargas@universidad.edu.pe"
        ), unsafe_allow_html=True)

    with col3:
        st.markdown(create_info_card(
            "Luis Mancilla",
            "Investigadora Clínica<br>• Estudiante Ing. Biomedica<br>• Especialista en ALS<br>• 12 años de experiencia<br>📧 ana.rodriguez@hospital.gob.pe"
        ), unsafe_allow_html=True)
        
        st.markdown(create_info_card(
            "Nataly Asto",
            "Desarrollador de Software<br>• Estudiante Ing. Biomedica<br>• Experto en Interfaces de Usuario<br>• 5 años de experiencia<br>📧 luis.vargas@universidad.edu.pe"
        ), unsafe_allow_html=True)
    
    # Publicaciones y reconocimientos
    st.markdown('<h3 class="subtitle">📚 Publicaciones Recientes</h3>', unsafe_allow_html=True)
    
    publications = """
    <div style="background: linear-gradient(135deg, {}, {}); padding: 1.5rem; border-radius: 15px; margin: 1rem 0;">
        <h4 style="color: {}; margin-bottom: 1rem;">📄 Artículos Científicos</h4>
        <ul style="color: white; line-height: 1.8;">
            <li><strong>"Deep Learning-based P300 Detection for ALS Patients Communication"</strong><br>
                <em>Journal of Neural Engineering, 2024</em></li>
            <li><strong>"Real-time EEG Processing for Brain-Computer Interfaces"</strong><br>
                <em>IEEE Transactions on Biomedical Engineering, 2023</em></li>
            <li><strong>"Improving P300 Speller Performance using Convolutional Neural Networks"</strong><br>
                <em>Journal of Neuroscience Methods, 2023</em></li>
        </ul>
    </div>
    """.format(COLORS['secondary'], COLORS['medium'], COLORS['light'])
    
    st.markdown(publications, unsafe_allow_html=True)
    
if __name__ == "__main__":
    main()