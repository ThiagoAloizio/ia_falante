import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
from ultralytics import YOLO
from google import genai
import time
import asyncio
import edge_tts
import os
import base64
from queue import Queue
from streamlit.runtime.scriptrunner import get_script_run_ctx

# Configuração da página do Streamlit
st.set_page_config(page_title="IA de Boas-Vindas Contextual", layout="wide")
st.title("🤖 IA de Boas-Vindas Contextual com YOLOv8 e Gemini")

# Silencia logs do OpenCV
os.environ["QT_LOGGING_RULES"] = "*.debug=false;*.info=false;*.warning=false"

# 1. Recursos Globais em Cache Estático Persistente
@st.cache_resource
def obter_recursos_globais():
    return Queue(), set()

objeto_queue, memoria_global_objetos = obter_recursos_globais()

# 2. Inicialização das variáveis de interface
if "texto_ia" not in st.session_state:
    st.session_state["texto_ia"] = ""
if "audio_html" not in st.session_state:
    st.session_state["audio_html"] = ""

# 3. Configuração Segura da API Key
try:
    MINHA_API_KEY = st.secrets["GEMINI_API_KEY"]
    client = genai.Client(api_key=MINHA_API_KEY)
except Exception:
    st.error("Erro: A chave 'GEMINI_API_KEY' não foi configurada nos Secrets do Streamlit.")
    st.stop()

@st.cache_resource
def load_yolo():
    return YOLO("yolov8n.pt")

yolo_model = load_yolo()

def obter_frase_criativa(lista_objetos):
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
    communicate = edge_tts.Communicate(texto, "pt-BR-FranciscaNeural")
    await communicate.save("resposta_web.mp3")

def transformar_audio_em_html(caminho_arquivo):
    with open(caminho_arquivo, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    return f"""
    <audio autoplay style="display:none;">
        <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
    </audio>
    """

# 4. Classe Processadora de Vídeo Corrigida para Sincronizar com a UI
class VideoProcessor(VideoProcessorBase):
    def __init__(self, queue, memoria_objetos, ctx):
        self.queue = queue
        self.memoria_objetos = memoria_objetos
        self.ctx = ctx  # Guarda o contexto de execução da página web
        self.tempo_visto_com_objetos = 0
        self.ultimo_tempo = time.time()

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        
        agora = time.time()
        dt = agora - self.ultimo_tempo
        self.ultimo_tempo = agora

        results = yolo_model(img, verbose=False)
        objetos_no_frame = set()

        for r in results:
            img = r.plot()
            for box in r.boxes:
                nome_objeto = yolo_model.names[int(box.cls)]
                objetos_no_frame.add(nome_objeto)

        itens_reais = [obj for obj in objetos_no_frame if obj != "person"]
        itens_novos = [item for item in itens_reais if item not in self.memoria_objetos]

        if itens_novos:
            self.tempo_visto_com_objetos += dt
            if self.tempo_visto_com_objetos > 2.0:
                self.memoria_objetos.update(itens_novos)
                self.queue.put(itens_novos)
                self.tempo_visto_com_objetos = 0
                
                # Força o Streamlit a atualizar a interface na hora usando o contexto salvo
                if self.ctx:
                    from streamlit.runtime.scriptrunner import ScriptRunner
                    ScriptRunner(self.ctx).request_rerun()
        else:
            self.tempo_visto_com_objetos = 0

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# 5. Interface Gráfica Web
col1, col2 = st.columns(2)

# Captura o contexto atual da sessão da interface ativa
contexto_atual = get_script_run_ctx()

with col1:
    st.subheader("📷 Feed de Vídeo em Tempo Real")
    webrtc_streamer(
        key="ia-boas-vindas",
        video_processor_factory=lambda: VideoProcessor(objeto_queue, memoria_global_objetos, contexto_atual),
        rtc_configuration={"iceServers": [{"urls": "stun:://google.com"}]},
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

    # 6. Monitor de Fila Síncrono (Processa imediatamente após o request_rerun da câmera)
    if not objeto_queue.empty():
        itens_para_processar = objeto_queue.get()
        
        with st.spinner("IA Pensando em uma interação..."):
            frase = obter_frase_criativa(itens_para_processar)
            st.session_state["texto_ia"] = ""  # Força limpeza visual
            st.session_state["texto_ia"] = frase
            
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(gerar_audio_edge(frase))
            loop.close()
            
            if os.path.exists("resposta_web.mp3"):
                st.session_state["audio_html"] = transformar_audio_em_html("resposta_web.mp3")
        st.rerun()

    # Renderiza o texto do Gemini na interface
    if st.session_state["texto_ia"]:
        st.info(st.session_state["texto_ia"])
    else:
        st.write("Aguardando detecção de novos objetos...")

    # Se houver áudio pendente, ele é executado pelo navegador
    if st.session_state["audio_html"]:
        st.markdown(st.session_state["audio_html"], unsafe_allow_html=True)
        st.session_state["audio_html"] = ""
