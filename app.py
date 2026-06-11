import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
from ultralytics import YOLO
from google import genai
import time
import asyncio
import edge_tts
import os
from queue import Queue

# Configuração da página do Streamlit
st.set_page_config(page_title="IA de Boas-Vindas Contextual", layout="wide")
st.title("🤖 IA de Boas-Vindas Contextual com YOLOv8 e Gemini")

# Silencia logs do OpenCV
os.environ["QT_LOGGING_RULES"] = "*.debug=false;*.info=false;*.warning=false"

# 1. Recursos Globais Estáticos Persistentes em Cache
@st.cache_resource
def obter_memoria_global():
    # Inicializa uma única vez a memória de objetos e a fila de comunicação
    return set(), Queue()

memoria_global_objetos, objeto_queue = obter_memoria_global()

# Inicialização estável das variáveis de controle na sessão
if "texto_ia" not in st.session_state:
    st.session_state["texto_ia"] = ""
if "tocar_audio" not in st.session_state:
    st.session_state["tocar_audio"] = False

# 2. Configuração Segura da API Key
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
    Uma pessoa acabou de chegar trazendo os seguintes objetos comuns: {lista_objetos}.
    Aja de forma natural, seja caloroso e interaja automaticamente sobre esses objetos com o usuário.
    Não use listas, não diga "foi detectado". Fale como um ser humano simpático recebendo um amigo em casa.
    Regra crucial: Faça um comentário detalhado, desenvolva bem o assunto sobre os itens trazidos e conclua com uma afirmação acolhedora. Escreva um parágrafo completo. Devolva APENAS o texto a ser falado.
    Proibição: Não faça nenhuma pergunta no final e não utilize pontos de interrogação.
    """
    tentativas = 3
    for i in range(tentativas):
        try:
            resposta = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt
            )
            return resposta.text.strip()
        except Exception as e:
            if i == tentativas - 1:
                return "Olá! Que bom que você chegou. Seja bem-vindo de volta! Deixe suas coisas na entrada e fique à vontade para descansar."
            time.sleep(1.5)

async def gerar_audio_edge(texto, nome_arquivo):
    communicate = edge_tts.Communicate(texto, "pt-BR-FranciscaNeural")
    await communicate.save(nome_arquivo)

# 3. Classe Processadora Nativa do WebRTC (Sem dependência de contexto de UI)
class VideoProcessor(VideoProcessorBase):
    def __init__(self, memoria_objetos, queue_comunicacao):
        self.memoria_objetos = memoria_objetos
        self.queue_comunicacao = queue_comunicacao
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

        # Filtra pessoas e foca nos objetos trazidos
        itens_reais = [obj for obj in objetos_no_frame if obj != "person"]
        itens_novos = [item for item in itens_reais if item not in self.memoria_objetos]

        if itens_novos:
            self.tempo_visto_com_objetos += dt
            # Se o objeto persistir por mais de 1.5 segundos (evita falsos positivos rápidos)
            if self.tempo_visto_com_objetos > 1.5:
                self.memoria_objetos.update(itens_novos)
                self.queue_comunicacao.put(itens_novos)
                self.tempo_visto_com_objetos = 0
        else:
            self.tempo_visto_com_objetos = 0

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# 4. Interface Gráfica Web (Layout de duas colunas)
col1, col2 = st.columns(2)

with col1:
    st.subheader("📷 Feed de Vídeo em Tempo Real")
    webrtc_streamer(
        key="ia-boas-vindas",
        video_processor_factory=lambda: VideoProcessor(memoria_global_objetos, objeto_queue),
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )

with col2:
    st.subheader("💬 Interação da Inteligência Artificial")
    
    if st.button("🔄 Limpar Memória de Objetos"):
        memoria_global_objetos.clear()
        st.session_state["texto_ia"] = ""
        st.session_state["tocar_audio"] = False
        while not objeto_queue.empty():
            objeto_queue.get()
        if os.path.exists("resposta_estavel.mp3"):
            try: os.remove("resposta_estavel.mp3")
            except: pass
        st.success("Memória limpa!")
        st.rerun()

    # Fragmento de atualização /pooling para verificar se a fila recebeu dados da câmera
    # Sem travar o vídeo
    if not objeto_queue.empty():
        itens_detectados = objeto_queue.get()
        
        with st.spinner("IA Pensando em uma interação..."):
            frase = obter_frase_criativa(itens_detectados)
            st.session_state["texto_ia"] = frase
            
            # Timestamp evita conflito de arquivos idênticos na nuvem
            nome_arquivo = f"resposta_{int(time.time())}.mp3"
            
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(gerar_audio_edge(frase, nome_arquivo))
            loop.close()
            
            st.session_state["arquivo_audio"] = nome_arquivo
            st.session_state["tocar_audio"] = True

    # Renderiza o texto gerado
    if st.session_state["texto_ia"]:
        st.info(st.session_state["texto_ia"])
    else:
        st.write("Aguardando detecção de novos objetos no vídeo...")

    # Se houver áudio pronto, renderiza o player nativo
    if st.session_state["tocar_audio"] and "arquivo_audio" in st.session_state:
        if os.path.exists(st.session_state["arquivo_audio"]):
            st.audio(st.session_state["arquivo_audio"], format="audio/mp3", autoplay=True)
            # Desativa para não entrar em loop infinito de áudio ao redesenhar a página
            st.session_state["tocar_audio"] = False 

    # Pequena automação para checar a fila a cada 2 segundos caso esteja vazia
    if objeto_queue.empty():
        time.sleep(2)
        st.rerun()
