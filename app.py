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
import threading

# Configuração da página do Streamlit
st.set_page_config(page_title="IA de Boas-Vindas Contextual", layout="wide")
st.title("🤖 IA de Boas-Vindas Contextual com YOLOv8 e Gemini")

# Silencia logs do OpenCV
os.environ["QT_LOGGING_RULES"] = "*.debug=false;*.info=false;*.warning=false"

# 1. Recursos Estáticos Persistentes para controle de falas
@st.cache_resource
def obter_memoria_global():
    return set()  # Guarda o histórico para NUNCA repetir o mesmo objeto no dia

memoria_global_objetos = obter_memoria_global()

# Inicialização das variáveis normais de interface
if "texto_ia" not in st.session_state:
    st.session_state["texto_ia"] = ""
if "audio_html" not in st.session_state:
    st.session_state["audio_html"] = ""

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
    Uma pessoa acabou de chegar trazendo os seguintes objetos inéditos que você ainda não tinha visto hoje: {lista_objetos}.
    Aja de forma natural, seja caloroso e interaja automaticamente sobre esses novos objetos com o usuário.
    Não use listas, não diga "foi detectado". Fale como um ser humano simpático recebendo um amigo em casa.
    Regra crucial: Faça um comentário detalhado, desenvolva bem o assunto sobre os itens trazidos e conclua com uma afirmação acolhedora. Escreva um parágrafo completo. Devolva APENAS o texto a ser falado.
    Proibição: Não faça nenhuma pergunta no final e não utilize pontos de interrogação.
    """
    try:
        resposta = client.models.generate_content(model='gemini-2.5-flash', contents=prompt)
        return resposta.text.strip()
    except Exception:
        return "Olá! Que bom que você chegou. Seja bem-vindo de volta! Deixe suas coisas na entrada e fique à vontade para descansar."

async def gerar_audio_edge(texto, nome_arquivo):
    communicate = edge_tts.Communicate(texto, "pt-BR-FranciscaNeural")
    await communicate.save(nome_arquivo)

def transformar_audio_em_html(caminho_arquivo):
    if not os.path.exists(caminho_arquivo):
        return ""
    with open(caminho_arquivo, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    return f"""
    <audio autoplay style="display:none;">
        <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
    </audio>
    """

def pipeline_processamento_ia(itens):
    """Executa a geração de inteligência artificial em segundo plano"""
    try:
        frase = obter_frase_criativa(itens)
        nome_arquivo = f"resposta_{int(time.time())}.mp3"
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(gerar_audio_edge(frase, nome_arquivo))
        loop.close()
        
        tag_html = transformar_audio_em_html(nome_arquivo)
        
        # Salva o resultado direto nas variáveis de sessão para exibição segura
        st.session_state["texto_ia"] = frase
        st.session_state["audio_html"] = tag_html
        
        if os.path.exists(nome_arquivo):
            try: os.remove(nome_arquivo)
            except: pass
    except Exception as e:
        print(f"[Erro Background]: {e}")

# 3. Classe Processadora Nativa do WebRTC (Thread-safe através de atributos de instância)
class VideoProcessor(VideoProcessorBase):
    def __init__(self, memoria_objetos):
        self.memoria_objetos = memoria_objetos
        self.tempo_visto_com_objetos = 0
        self.ultimo_tempo = time.time()
        self.novos_itens_compartilhados = None  # Canal limpo de comunicação externa

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
                # Guarda o dado diretamente no atributo da classe, visível para a interface
                self.novos_itens_compartilhados = itens_novos
                self.tempo_visto_com_objetos = 0
        else:
            self.tempo_visto_com_objetos = 0

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# 4. Interface Gráfica Web (Layout de duas colunas)
col1, col2 = st.columns(2)

with col1:
    st.subheader("📷 Feed de Vídeo em Tempo Real")
    # Capturamos o contexto do WebRTC streamer na variável ctx
    ctx = webrtc_streamer(
        key="ia-boas-vindas",
        video_processor_factory=lambda: VideoProcessor(memoria_global_objetos),
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

    # Contêiner onde os textos e avisos serão desenhados dinamicamente
    area_resposta_ia = st.empty()

    # 5. O SEGREDO DO FUNCIONAMENTO: Varredura contínua e nativa do estado da câmera
    if ctx.video_processor:
        # Verifica diretamente se a propriedade interna da instância da câmera mudou
        if ctx.video_processor.novos_itens_compartilhados is not None:
            itens_detectados = ctx.video_processor.novos_itens_compartilhados
            ctx.video_processor.novos_itens_compartilhados = None  # Limpa o gatilho na hora
            
            with area_resposta_ia.container():
                with st.spinner("IA Pensando em uma interação..."):
                    # Executa a geração em uma thread para não engasgar os quadros da webcam
                    t = threading.Thread(target=pipeline_processamento_ia, args=(itens_detectados,))
                    t.start()
                    t.join()  # Sincroniza o fim do processamento do texto/áudio antes de desenhar a tela
            st.rerun()

    # Desenha o resultado estável na interface web
    with area_resposta_ia.container():
        if st.session_state["texto_ia"]:
            st.info(st.session_state["texto_ia"])
        else:
            st.write("Aguardando detecção de novos objetos...")

        if st.session_state["audio_html"]:
            st.markdown(st.session_state["audio_html"], unsafe_allow_html=True)
            st.session_state["audio_html"] = ""  # Consome a tag

# 6. Atualizador ultra leve de interface (Diz para a tela ler o ctx da câmera a cada 1 segundo)
from streamlit_autorefresh import st_autorefresh
st_autorefresh(interval=1000, key="atualizador_de_interface_nuvem")
