import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
from ultralytics import YOLO
from google import genai
import time
import os
from queue import Queue
from gtts import gTTS
from streamlit_autorefresh import st_autorefresh
import random

# Configuração da página do Streamlit
st.set_page_config(page_title="IA de Boas-Vindas Contextual", layout="wide")
st.title("🤖 IA de Boas-Vindas Contextual com YOLOv8 e Gemini")

# Silencia logs do OpenCV
os.environ["QT_LOGGING_RULES"] = "*.debug=false;*.info=false;*.warning=false"

# 1. Recursos Globais Estáticos Persistentes em Cache
@st.cache_resource
def obter_memoria_global():
    return set(), Queue()

memoria_global_objetos, objeto_queue = obter_memoria_global()

# Inicialização estável das variáveis de controle na sessão
if "texto_ia" not in st.session_state:
    st.session_state["texto_ia"] = ""
if "tocar_audio" not in st.session_state:
    st.session_state["tocar_audio"] = False
if "arquivo_audio" not in st.session_state:
    st.session_state["arquivo_audio"] = ""
if "processando" not in st.session_state:
    st.session_state["processando"] = False

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
    # Dicionário de tradução para os objetos mais comuns do YOLOv8 em uma casa
    traducoes = {
        "person": "pessoa",
        "bicycle": "bicicleta",
        "car": "carro",
        "motorcycle": "moto",
        "backpack": "mochila",
        "umbrella": "guarda-chuva",
        "handbag": "bolsa",
        "tie": "gravata",
        "suitcase": "mala",
        "bottle": "garrafa",
        "wine glass": "taça de vinho",
        "cup": "copo",
        "fork": "garfo",
        "knife": "faca",
        "spoon": "colher",
        "bowl": "tigela",
        "banana": "banana",
        "apple": "maçã",
        "sandwich": "sanduíche",
        "orange": "laranja",
        "broccoli": "brócolis",
        "carrot": "cenoura",
        "hot dog": "cachorro-quente",
        "pizza": "pizza",
        "donut": "donut",
        "cake": "bolo",
        "chair": "cadeira",
        "couch": "sofá",
        "potted plant": "vaso de planta",
        "bed": "cama",
        "dining table": "mesa de jantar",
        "tv": "televisão",
        "laptop": "notebook",
        "mouse": "mouse",
        "remote": "controle remoto",
        "keyboard": "teclado",
        "cell phone": "celular",
        "microwave": "micro-ondas",
        "oven": "forno",
        "toaster": "torradeira",
        "sink": "pia",
        "refrigerator": "geladeira",
        "book": "livro",
        "clock": "relógio",
        "vase": "vaso",
        "scissors": "tesoura",
        "teddy bear": "ursinho de pelúcia",
        "hair drier": "secador de cabelo",
        "toothbrush": "escova de dentes"
    }

    # Traduz os objetos detectados. Se não estiver no dicionário, mantém o nome original
    itens_traduzidos = [traducoes.get(obj, obj) for obj in lista_objetos]
    itens_formatados = ", ".join(itens_traduzidos)
    
    # Banco de frases locais (Fallback) totalmente em português
    saudacoes_locais = [
        f"Seja muito bem-vindo de volta! Que ótimo que você chegou trazendo seu {itens_formatados}. Pode deixar tudo na entrada e ir descansar um pouco.",
        f"Olha só quem chegou! Vejo que trouxe um {itens_formatados} com você hoje. Sinta-se em casa, tire os sapatos e aproveite o resto do seu dia.",
        f"Olá, que bom ver você! Muito interessante você estar com seu {itens_formatados} agora. Entre e fique totalmente à vontade."
    ]
    
    prompt = f"""
    Você é uma inteligência artificial de boas-vindas instalada na entrada de uma casa.
    Uma pessoa acabou de chegar trazendo os seguintes objetos comuns: {itens_formatados}.
    Aja de forma natural, seja caloroso e interaja automaticamente sobre esses objetos com o usuário.
    Não use listas, não diga "foi detectado". Fale como um ser humano simpático recebendo um amigo em casa.
    Regra crucial: Faça um comentário detalhado, desenvolva bem o assunto sobre os itens trazidos e conclua com uma afirmação acolhedora. Escreva um parágrafo completo. Devolva APENAS o texto a ser falado.
    Proibição: Não faça nenhuma pergunta no final e não utilize pontos de interrogação.
    """
    
    try:
        resposta = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt
        )
        if resposta.text:
            return resposta.text.strip()
        elif hasattr(resposta, 'candidates') and resposta.candidates:
            return resposta.candidates[0].content.parts[0].text.strip()
            
    except Exception as e:
        # Se a API falhar por cota, o fallback local agora usará os itens traduzidos!
        st.sidebar.warning("⚠️ API do Gemini sem cota temporariamente. Usando gerador local traduzido.")
        return random.choice(saudacoes_locais)

def gerar_audio_gtts(texto, nome_arquivo):
    try:
        tts = gTTS(text=texto, lang='pt', tld='com.br')
        tts.save(nome_arquivo)
        return True
    except Exception as e:
        st.error(f"Erro ao gerar áudio: {e}")
        return False

# =====================================================================
# 3. Classe Processadora Nativa do WebRTC (Com Tradução na Tela)
# =====================================================================
class VideoProcessor(VideoProcessorBase):
    def __init__(self, memoria_objetos, queue_comunicacao):
        self.memoria_objetos = memoria_objetos
        self.queue_comunicacao = queue_comunicacao
        self.tempo_visto_com_objetos = 0
        self.ultimo_tempo = time.time()
        self.ultima_requisicao_ia = 0

        # Dicionário idêntico para garantir a tradução visual nos frames da câmera
        self.traducoes_visuais = {
            "person": "pessoa", "bicycle": "bicicleta", "car": "carro", "motorcycle": "moto",
            "backpack": "mochila", "umbrella": "guarda-chuva", "handbag": "bolsa", "tie": "gravata",
            "suitcase": "mala", "bottle": "garrafa", "wine glass": "taça de vinho", "cup": "copo",
            "fork": "garfo", "knife": "faca", "spoon": "colher", "bowl": "tigela", "banana": "banana",
            "apple": "maçã", "sandwich": "sanduíche", "orange": "laranja", "broccoli": "brócolis",
            "carrot": "cenoura", "hot dog": "cachorro-quente", "pizza": "pizza", "donut": "donut",
            "cake": "bolo", "chair": "cadeira", "couch": "sofá", "potted plant": "vaso de planta",
            "bed": "cama", "dining table": "mesa de jantar", "tv": "televisão", "laptop": "notebook",
            "mouse": "mouse", "remote": "controle remoto", "keyboard": "teclado", "cell phone": "celular",
            "microwave": "micro-ondas", "oven": "forno", "toaster": "torradeira", "sink": "pia",
            "refrigerator": "geladeira", "book": "livro", "clock": "relógio", "vase": "vaso",
            "scissors": "tesoura", "teddy bear": "ursinho de pelúcia", "hair drier": "secador de cabelo",
            "toothbrush": "escova de dentes"
        }

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        import cv2 # Garante que o OpenCV está disponível para desenho manual se necessário
        img = frame.to_ndarray(format="bgr24")
        
        agora = time.time()
        dt = agora - self.ultimo_tempo
        self.ultimo_tempo = agora

        results = yolo_model(img, verbose=False)
        objetos_no_frame = set()

        for r in results:
            # Em vez de usar o r.plot() padrão que fixa o inglês, 
            # vamos alterar temporariamente os nomes mapeados no modelo para este frame
            nomes_originais = r.names.copy()
            
            # Sobrescreve as classes detectadas com os nomes traduzidos para exibição na caixa
            for idx, nome_en in r.names.items():
                if nome_en in self.traducoes_visuais:
                    r.names[idx] = self.traducoes_visuais[nome_en]
            
            # Agora o plot() vai desenhar com os nomes em português!
            img = r.plot()
            
            # Restaura os nomes originais no modelo para não quebrar a lógica interna do YOLO
            r.names = nomes_originais

            # Coleta os nomes originais em inglês para a lógica da fila e do gTTS
            for box in r.boxes:
                nome_objeto = yolo_model.names[int(box.cls)]
                objetos_no_frame.add(nome_objeto)

        # 1. Filtra pessoas e foca nos objetos trazidos
        # ADICIONAMOS "fire hydrant" na lista de exclusão para o YOLO ignorar esse erro!
        itens_reais = [obj for obj in objetos_no_frame if obj not in ["person", "fire hydrant"]]

        if itens_reais:
            self.tempo_visto_com_objetos += dt
            if self.tempo_visto_com_objetos > 2.0:
                if (agora - self.ultima_requisicao_ia > 15):
                    self.queue_comunicacao.put(itens_reais)
                    self.ultima_requisicao_ia = agora
                self.tempo_visto_com_objetos = 0
        else:
            self.tempo_visto_com_objetos = 0

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# =====================================================================
# 4. Interface Gráfica Web (Layout de duas colunas)
# =====================================================================

# Se a IA estiver tocando áudio, aumentamos o tempo do refresh para 12 segundos
# Isso dá tempo de a frase ser totalmente falada sem interrupções!
# Se não estiver tocando, mantém em 3 segundos para monitorar a câmera de perto.
tempo_refresh = 12000 if st.session_state["tocar_audio"] else 3000
st_autorefresh(interval=tempo_refresh, key="atualizador_fila")

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
        st.session_state["arquivo_audio"] = ""
        st.session_state["processando"] = False
        while not objeto_queue.empty():
            objeto_queue.get()
        st.success("Memória limpa!")
        st.rerun()

    # Processa a fila de forma segura
    if not objeto_queue.empty() and not st.session_state["processando"]:
        st.session_state["processando"] = True
        itens_detectados = objeto_queue.get()
        
        with st.spinner("IA Pensando em uma interação..."):
            frase = obter_frase_criativa(itens_detectados)
            st.session_state["texto_ia"] = frase
            
            nome_arquivo = f"resposta_{int(time.time())}.mp3"
            if gerar_audio_gtts(frase, nome_arquivo):
                st.session_state["arquivo_audio"] = nome_arquivo
                st.session_state["tocar_audio"] = True
        
        st.session_state["processando"] = False

    # Renderiza o texto gerado da IA
    if st.session_state["texto_ia"]:
        st.info(st.session_state["texto_ia"])
    else:
        st.write("Aguardando detecção de novos objetos no vídeo...")

    # Se houver áudio pronto, renderiza o player nativo com autoplay
    if st.session_state["arquivo_audio"]:
        if os.path.exists(st.session_state["arquivo_audio"]):
            st.audio(st.session_state["arquivo_audio"], format="audio/mp3", autoplay=True)
            
            # ATENÇÃO: Removemos o gatilho que resetava o tocar_audio imediatamente aqui,
            # permitindo que a variável controle o tempo do 'tempo_refresh' lá no topo.
            # O próprio gTTS cria arquivos únicos por timestamp, então não há risco de loop infinito.
