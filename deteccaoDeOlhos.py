import cv2

# Abre o classificador pelo qual você deseja identificar a imagem(Carros, Pessoas, Animais etc)
classificadorFace = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')

# Abre o classificador de olhos
classificadorOlhos = cv2.CascadeClassifier('cascades/haarcascade_eye.xml')

# Abre a imagem da qual você deseja fazer a verificação e converte a mesma para cinza.
imagem = cv2.imread('pessoas/pessoas4.jpg')
imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)


#   scaleFactor= é utilizado para detectar faces menores nas fotos, e assim vai regulando os parâmetros.
# ele varia entre (1.01 e 1.3), por default ele já vem com 1.1!
#   minNeighbors= valores maiores: menos detecções mas, maior qualidade
#                 valores baixos:  mais detecções mas, qualidade menor
#   minSize= É o menor numero de pixels x pixels que você deseja que o algoritmo reconheça.
faceDetectadas = classificadorFace.detectMultiScale(imagemCinza, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30))

for (x, y, l, a) in faceDetectadas:
    imagem = cv2.rectangle(imagem, (x, y), (x + l, y + a), (0, 0, 255), 2)
    # Região para desenhar a parte dos olhos, das quais você deseja.
    regiao = imagem[y:y + a, x:x + l]
    # Converte a região para cinza, para ficar mais fácil a busca.
    regiaoCinzaOlho = cv2.cvtColor(regiao, cv2.COLOR_BGR2GRAY)
    # Faz a mesma coisa que faz na face, só que com os olhos, para você poder aprimorar e aperfeiçoar a ferramenta.
    olhosDetectados = classificadorOlhos.detectMultiScale(regiaoCinzaOlho, scaleFactor=1.19, minNeighbors=3)
    print(olhosDetectados)
    for (ox, oy, ol, oa) in olhosDetectados:
        # Faz o segundo for, para detectar os olhos.
        cv2.rectangle(regiao, (ox, oy), (ox + ol, oy + oa), (255, 0, 255), 2)

# Print da img na tela, com o primeiro sendo o nome da tela.
cv2.imshow('Faces e olhos detectados', imagem)
cv2.waitKey()   # Sai da tela apenas com um apertar de qualquer tecla
