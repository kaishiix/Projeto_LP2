import cv2
import os
import numpy as np

# --- Constantes para Segmentação ---
# Rosa Choque em BGR (Blue, Green, Red)
NEW_BACKGROUND_COLOR = (255, 0, 255) 

# --- Função de Pré-Processamento (Segmentação e Troca de Fundo) ---
def preprocess_and_replace_background(img):
    """
    Tenta segmentar o objeto principal (assumido ser a moto) usando GrabCut
    e troca o fundo para Rosa Choque.
    Retorna a imagem processada.
    """
    h, w = img.shape[:2]
    
    # GrabCut requer uma máscara inicial, buffers e um retângulo inicial
    mask = np.zeros(img.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    
    # Retângulo inicial (exclui uma pequena margem para ajudar o GrabCut)
    rect = (1, 1, w - 2, h - 2) 
    
    try:
        # 1. Aplicar GrabCut
        cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
        
        # 2. Criar máscara binária (onde 1 e 3 são foreground)
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        
        # 3. Criar a imagem de fundo Rosa Choque
        processed_img = np.full(img.shape, NEW_BACKGROUND_COLOR, dtype=np.uint8)
        
        # 4. Combinar o objeto (moto) com o novo fundo
        processed_img = img * mask2[:, :, np.newaxis] + processed_img * (1 - mask2[:, :, np.newaxis])
        
        return processed_img

    except Exception as e:
        print(f"⚠️ Aviso: Falha na segmentação GrabCut: {e}. Usando imagem original.")
        return img


# --- Função para Estimar o Nome da Cor (CORRIGIDA E OTIMIZADA) ---
# --- Função para Estimar o Nome da Cor (CORRIGIDA PARA VERDE/AMARELO) ---
def get_dominant_color_name(bgr_color):
    b, g, r = [int(c) for c in bgr_color]
    total = b + g + r
    max_val = max(r, g, b)
    print(b, g, r)
    # --- Cores neutras ---
    if total < 100:
        return "preto"
    
    if r > 230 and g > 230 and b > 230:
        return "branco"
    
    if abs(r - g) < 20 and abs(r - b) < 20 and abs(g - b) < 20:
        if total > 400:
            return "cinza_claro"
        else:
            return "cinza_escuro"

    # --- Amarelo ---
    if r > 180 and g > 180 and b < 100:
        return "amarelo"

    # --- Laranja ---
    if r > 180 and 100 < g < 180 and b < 100:
        return "laranja"

    # --- Marrom ---
    if r > 100 and g > 40 and g < 100 and b < 80 and total < 350:
        return "marrom"

    # --- Vermelha (ajuste para captar tons mais reais como 207, 53, 182) ---
    if r > 150 and r > g + 30 and r > b + 15:
        return "vermelha"

    # --- Verde Puro ---
    if g > 150 and g > r + 50 and g > b + 50:
        return "verde"

    # --- Verde Escuro / Apagado (ex: 100, 76, 100) ---
    if 60 < g < 110 and abs(r - g) < 40 and abs(b - g) < 40:
        return "verde_escuro"

    # --- Azul ---
    if b > 150 and b > r + 30 and b > g + 50:
        return "azul"
    # --- Roxo ---
    if r > 120 and b > 120 and g < 100:
        return "roxo"

    return "indefinida"



# --- Configuração do Modelo ---
config_file = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model = 'frozen_inference_graph.pb'

# Garante que os arquivos do modelo existam
if not os.path.exists(frozen_model) or not os.path.exists(config_file):
    print("ERRO: Arquivos do modelo não encontrados. Certifique-se de que .pb e .pbtxt estão no diretório.")
    exit()

model = cv2.dnn_DetectionModel(frozen_model, config_file)

# --- Carregar Rótulos (Labels) ---
classLabels = []
filename = 'labels.txt'
try:
    with open(filename, 'rt') as spt:
        classLabels = spt.read().rstrip('\n').split('\n')
except FileNotFoundError:
    print(f"Erro: O arquivo de rótulos '{filename}' não foi encontrado.")
    exit()
    
MOTORCYCLE_CLASS_INDEX = 4 

# --- Configurações de Entrada do Modelo ---
model.setInputSize(320, 320)
model.setInputScale(1.0/127.5)
model.setInputMean((127.5, 127.5, 127.5))
model.setInputSwapRB(True)

# --- Carregar Imagem ---
image_path = 'avi.jpg'
img = cv2.imread(image_path)

if img is None:
    print(f"Erro: Não foi possível carregar a imagem em '{image_path}'.")
    exit()
    
# ------------------------------------------------------------------
# --- PRÉ-PROCESSAMENTO: CHAMA A FUNÇÃO DE TROCA DE FUNDO AQUI ---
# ------------------------------------------------------------------
img_processed = preprocess_and_replace_background(img.copy())


# --- Detecção ---
# A detecção é feita na imagem original/copiada para garantir a BBOX correta.
classIndex, confidence, bbox = model.detect(img, confThreshold=0.5) 

motos_detectadas_info = [] 
font = cv2.FONT_HERSHEY_PLAIN

# Verifica se foram detectados objetos
if len(classIndex) > 0:
    for classInd, conf, boxes in zip(classIndex.flatten(), confidence.flatten(), bbox):
        
        # Desenha na imagem original (img)
        cv2.rectangle(img, boxes, (255, 0, 0), 2)
        label = classLabels[classInd-1]
        
        if classInd == MOTORCYCLE_CLASS_INDEX:
            
            # --- Classificação de Cor ---
            x, y, w, h = boxes
            # Garante que as coordenadas da caixa sejam válidas
            x = max(0, x)
            y = max(0, y)
            w = min(img.shape[1] - x, w)
            h = min(img.shape[0] - y, h)
            
            # Usa a imagem PROCESSADA (fundo rosa choque) para a análise de cor
            moto_region = img_processed[y:y+h, x:x+w]
            # Média BGR (Mantido np.uint16 para prevenir overflow na média)
            average_bgr = moto_region.mean(axis=(0, 1)).astype(np.uint16) 
            
            # Obtém os nomes de cor
            color_name_folder = get_dominant_color_name(average_bgr)
            color_name_display = color_name_folder.replace('_', ' ').title()
            
            motos_detectadas_info.append({
                'conf': conf, 
                'color_display': color_name_display, 
                'color_folder': color_name_folder, 
                'bgr': average_bgr
            })

            # Atualiza o rótulo na imagem com a cor
            display_text = f"{label} ({color_name_display})"
            cv2.putText(img, display_text, (x + 10, y + 40), font, fontScale = 2, color=(0, 255, 0), thickness=3)
        
        else:
            # Rótulo para outros objetos
            cv2.putText(img, label, (boxes[0] + 10, boxes[1] + 40), font, fontScale = 2, color=(0, 255, 0), thickness=3)


# --- Criar Pastas e Salvar Resultados ---

FOLDER_NAO_MOTO = 'nao_motos'
FILE_INFO_MOTO = 'motos_ok.txt'
FILE_INFO_NAO_MOTO = 'nao_moto.txt'
OUTPUT_IMAGE_NAME = os.path.basename(image_path) 

if motos_detectadas_info:
    # --- MOTO DETECTADA ---
    # PEGA O NOME DA COR DA PRIMEIRA MOTO PARA A PASTA
    first_moto_color = motos_detectadas_info[0]['color_folder']
    output_folder = f'moto_{first_moto_color}' # Nome da pasta: moto_cor_identificada
    info_filename = FILE_INFO_MOTO
    
    # Criar a pasta de destino
    os.makedirs(output_folder, exist_ok=True)
    
    # Salvar o arquivo de texto com as informações
    output_info_path = os.path.join(output_folder, info_filename)
    with open(output_info_path, 'w') as f:
        f.write(f"Motos detectadas em {image_path}:\n")
        
        for i, info in enumerate(motos_detectadas_info):
            color_bgr_str = f"({info['bgr'][0]}, {info['bgr'][1]}, {info['bgr'][2]})"
            f.write(f"  - Moto {i+1}: Cor Estimada: {info['color_display']} | Confiança: {info['conf']:.2f} | Média BGR: {color_bgr_str}\n")
            
    print(f"✅ Arquivo de info salvo em '{output_info_path}'.")
    
else:
    # --- NENHUMA MOTO DETECTADA ---
    output_folder = FOLDER_NAO_MOTO
    info_filename = FILE_INFO_NAO_MOTO
    
    # Criar a pasta de destino
    os.makedirs(output_folder, exist_ok=True)
    
    # Salvar o arquivo de texto
    output_info_path = os.path.join(output_folder, info_filename)
    with open(output_info_path, 'w') as f:
        f.write(f"Nenhuma moto detectada em {image_path}.")
        
    print(f"❌ Arquivo de info salvo em '{output_info_path}'.")

# --- Salvar Imagem Final (com as caixas de detecção) na Pasta ---
output_image_path = os.path.join(output_folder, f"DETECTED_{OUTPUT_IMAGE_NAME}")
cv2.imwrite(output_image_path, img)
print(f"🖼️ Imagem final com detecções salva em '{output_image_path}'.")

# --- Mostrar Imagem ---
cv2.imshow('Resultado da Detecção', img)
cv2.waitKey(0) 
cv2.destroyAllWindows()