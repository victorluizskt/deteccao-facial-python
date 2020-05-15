import cv2

# Abre o classificador pelo qual você deseja identificar a imagem(Carros, Pessoas, Animais etc)
classificador = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')

# Abre a imagem da qual você deseja fazer a verificação e converte a mesma para cinza.
imagem = cv2.imread('pessoas/pessoas4.jpg')
imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

#   scaleFactor= é utilizado para detectar faces menores nas fotos, e assim vai regulando os parâmetros.
# ele varia entre (1.01 e 1.3), por default ele já vem com 1.1!
#   minNeighbors= valores maiores: menos detecções mas, maior qualidade
#                 valores baixos:  mais detecções mas, qualidade menor
#   minSize= É o menor numero de pixels x pixels que você deseja que o algoritmo reconheça.
facesDetectadas = classificador.detectMultiScale(imagemCinza, scaleFactor=1.39, minNeighbors=5, minSize=(40, 40))

print(len(facesDetectadas))
print(facesDetectadas)

for (x, y, l, a) in facesDetectadas:
    print(x, y, l, a)
    # 1 Parametro: Pega a posição do x e y  2 Parametro: x + l = soma a largura + altura e desenha a reta, logo após
    # soma y + a e faz a borda vertical.
    # 3 Paramentro: Cor do quadrado      Ultimo parametro:Tamanho da borda
    cv2.rectangle(imagem, (x, y), (x + l, y + a), (10, 240, 80), 2)

cv2.imshow('Faces encontradas', imagem)
cv2.waitKey()
