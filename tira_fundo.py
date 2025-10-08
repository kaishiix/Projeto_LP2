import cv2
import numpy as np
import os

def preprocess_image_for_color_analysis(input_image_path, output_image_path):
    """
    Carrega uma imagem, tenta segmentar o objeto principal (assumido ser a moto)
    e troca o fundo para Ciano para uma melhor análise de cor.
    
    Args:
        input_image_path (str): Caminho para a imagem original (ex: 'test_image.png').
        output_image_path (str): Caminho para salvar a imagem processada.
    """
    
    # 1. Carregar a Imagem
    img = cv2.imread(input_image_path)
    if img is None:
        print(f"Erro: Não foi possível carregar a imagem em '{input_image_path}'.")
        return None

    # Definir a cor do novo fundo (Ciano) em BGR (Azul, Verde, Vermelho)
    # Ciano em BGR é (255, 255, 0)
    NEW_BACKGROUND_COLOR = (255, 255, 0)

    # 2. Inicializar o GrabCut (Segmentação de Fundo)
    
    # O GrabCut requer uma máscara inicial e um retângulo
    mask = np.zeros(img.shape[:2], np.uint8)
    
    # buffers para o GrabCut
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    
    # Define o retângulo inicial. 
    # GrabCut tentará encontrar o objeto dentro deste retângulo.
    # Exemplo: 10 pixels de margem em todos os lados.
    h, w = img.shape[:2]
    rect = (10, 10, w - 20, h - 20) 
    
    # **IMPORTANTE:** Este retângulo é uma ESTIMATIVA. 
    # Se a moto estiver na borda ou for muito pequena, você precisará ajustá-lo.
    
    try:
        # 3. Aplicar GrabCut
        cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
        
        # A máscara retorna valores 0, 1, 2, 3. 
        # Queremos que 1 (foreground) e 3 (provável foreground) sejam o objeto.
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        
        # 4. Criar a Imagem Processada
        
        # Crie uma imagem de fundo sólida com a cor Ciano
        processed_img = np.full(img.shape, NEW_BACKGROUND_COLOR, dtype=np.uint8)
        
        # Copia o objeto (moto) da imagem original para a imagem processada
        # Note que a multiplicação por 'mask2' garante que apenas o objeto seja mantido.
        processed_img = img * mask2[:, :, np.newaxis] + processed_img * (1 - mask2[:, :, np.newaxis])
        
        # 5. Salvar o Resultado
        cv2.imwrite(output_image_path, processed_img)
        print(f"✅ Imagem processada e fundo trocado para Ciano. Salva em: {output_image_path}")
        return output_image_path

    except Exception as e:
        print(f"Erro ao executar o GrabCut/Segmentação: {e}")
        # Em caso de falha, retorna a imagem original para não quebrar o pipeline
        cv2.imwrite(output_image_path, img) 
        print(f"⚠️ Falha na segmentação. A imagem original foi salva como: {output_image_path}")
        return output_image_path


if __name__ == '__main__':
    # Define o caminho da imagem de entrada e da imagem de saída (pré-processada)
    INPUT_FILE = 'test_image.png'
    OUTPUT_PROCESSED_FILE = 'test_image_preprocessed.png'
    
    # Executa o pré-processamento
    if os.path.exists(INPUT_FILE):
        preprocess_image_for_color_analysis(INPUT_FILE, OUTPUT_PROCESSED_FILE)
        
        # Opcional: Mostrar a imagem processada
        processed_img = cv2.imread(OUTPUT_PROCESSED_FILE)
        if processed_img is not None:
            cv2.imshow('Imagem Pre-Processada (Fundo Ciano)', processed_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    else:
        print(f"ERRO: Arquivo de entrada '{INPUT_FILE}' não encontrado. Crie-o ou ajuste o nome.")