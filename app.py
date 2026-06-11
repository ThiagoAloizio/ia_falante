import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
from ultralytics import YOLO
from google import genai
import time
import asyncio
import edge_tts
import os
import base64

# Configuração da página do Streamlit
st.set_page_config(page_title="IA de Boas-Vindas Contextual", layout="wide")
st.title("🤖 IA de Boas-Vindas Contextual com YOLOv8 e Gemini")

# Silencia logs do OpenCV
os.environ["QT_LOGGING_RULES"] = "*.debug=false;*.info=false;*.warning=false"

# 1. Recursos Globais em Cache (Fila de mensageria e memória de curto prazo)
@st.cache_resource
def obter_recursos_globais():
    from queue import Queue
    return Queue(), set(), {"tempo_visto": 0, "ultimo_tempo": time.time()}

objeto_queue, memoria_global_objetos, cronometro = obter_recursos_globais()

# 2. Inicialização das variáveis de interface
if "texto_ia" not in st.session_state:
    st.session_state["texto_ia"] = ""
if "audio_html" not in st.session_state:
    st.session_state["audio_html"] = ""

# 3. Configuração Segura da API Key (Puxando dos Secrets da Nuvem)
try:
    MINHA_API_KEY = st.secrets["GEMINI_API_KEY"]
    client = genai.Client(api_key=MINHA_API_KEY)
except Exception as e:
    st.error("Erro: A chave 'GEMINI_API_KEY' não foi configurada nos Secrets do Streamlit.")
    st.stop()

@st.cache_resource
def load_yolo():
    return YOLO("yolov8n.pt")

yolo_model = load_yolo()

def obter_frase_criativa(lista_objetos):
    """Solicita a interação para o Gemini com Retry automático"""
    prompt = f"""
    Você é uma inteligência artificial de boas-vindas instalada na entrada de uma casa.
    Uma pessoa acabou de chegar trazendo os seguintes objetos inéditos que você ainda não tinha visto hoje: {lista_objetos}.
    Aja de forma natural, seja caloroso e interaja automaticamente sobre esses novos objetos com o usuário.
    Não use listas, não diga "foi detectado". Fale como um ser humano simpático recebendo um amigo em casa.
    Regra crucial: Faça um comentário detalhado, desenvolva bem o assunto sobre os itens trazidos e conclua com uma afirmação acolhedora. Escreva um parágrafo completo. Devolva APENAS o texto a ser falado.
    Proibição: Não faça nenhuma pergunta no final e não utilize pontos de interrogação.
    """
    tentativas = 3
    for i in range(tentativas):
        try:
            resposta = client.models.generate_content(model='gemini-2.5-flash', contents=prompt)
            return resposta.text.strip()
        except Exception:
            if i == tentativas - 1:
                return "Olá! Que bom que você chegou. Seja bem-vindo de volta! Deixe suas coisas na entrada e fique à vontade para descansar."
            time.sleep(1.5)

async def gerar_audio_edge(texto):
    """Gera o arquivo temporário de áudio"""
    communicate = edge_tts.Communicate(texto, "pt-BR-FranciscaNeural")
    await communicate.save("resposta_web.mp3")

def transformar_audio_em_html(caminho_arquivo):
    """Codifica o áudio em Base64 para tocar nativamente no navegador do cliente"""
    with open(caminho_arquivo, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    return f"""
    <audio autoplay style="display:none;">
        <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
    </audio>
    """

# 4. Callback de processamento de imagem do WebRTC
def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")
    
    agora = time.time()
    dt = agora - cronometro["ultimo_tempo"]
    cronometro["ultimo_tempo"] = agora

    results = yolo_model(img, verbose=False)
    objetos_no_frame = set()

    for r in results:
        img = r.plot()
        for box in r.boxes:
            nome_objeto = yolo_model.names[int(box.cls)]
            objetos_no_frame.add(nome_objeto)

    itens_reais = [obj for obj in objetos_no_frame if obj != "person"]
    itens_novos = [item for item in itens_reais if item not in memoria_global_objetos]

    if itens_novos:
        cronometro["tempo_visto"] += dt
        if cronometro["tempo_visto"] > 2.0:
            memoria_global_objetos.update(itens_novos)
            objeto_queue.put(itens_novos)
            cronometro["tempo_visto"] = 0
    else:
        cronometro["tempo_visto"] = 0

    return av.VideoFrame.from_ndarray(img, format="bgr24")

# 5. Interface Gráfica Web
col1, col2 = st.columns(2)

with col1:
    st.subheader("📷 Feed de Vídeo em Tempo Real")
    webrtc_streamer(
        key="ia-boas-vindas",
        video_frame_callback=video_frame_callback,
        media_stream_constraints={"video": True, "audio": False},
    )

with col2:
    st.subheader("💬 Interação da Inteligência Artificial")
    
    if st.button("🔄 Limpar Memória de Objetos"):
        memoria_global_objetos.clear()
        st.session_state["texto_ia"] = ""
        st.session_state["audio_html"] = ""
        st.success("Memória de objetos limpa!")
        st.rerun()

    # 6. Escuta a fila e processa as chamadas de IA fora da thread de vídeo
    if not objeto_queue.empty():
        itens_para_processar = objeto_queue.get()
        
        with st.spinner("IA Pensando em uma interação..."):
            frase = obter_frase_criativa(itens_para_processar)
            st.session_state["texto_ia"] = frase
            
            # Executa o Edge-TTS de forma isolada
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(gerar_audio_edge(frase))
            loop.close()
            
            # Gera a tag HTML de reprodução do áudio
            if os.path.exists("resposta_web.mp3"):
                st.session_state["audio_html"] = transformar_audio_em_html("resposta_web.mp3")
        st.rerun()

    # Exibe o parágrafo de texto gerado
    if st.session_state["texto_ia"]:
        st.info(st.session_state["texto_ia"])
    else:
        st.write("Aguardando detecção de novos objetos...")

    # Se houver áudio pendente, ele é injetado e executado pelo navegador
    if st.session_state["audio_html"]:
        st.markdown(st.session_state["audio_html"], unsafe_allow_html=True)
        st.session_state["audio_html"] = ""  # Consome o áudio para evitar loops
