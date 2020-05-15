import cv2

classificadorGato = cv2.CascadeClassifier('cascades/haarcascade_frontalcatface.xml')

imagem = cv2.imread('outros/gato3.jpg')
imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

detectado = classificadorGato.detectMultiScale(imagemCinza, scaleFactor=1.05)

for (x, y, l, a) in detectado:
    imagem = cv2.rectangle(imagem, (x, y), (x + l, y + a), (0, 0, 255), 2)


cv2.imshow('Encontrado', imagem)
cv2.waitKey()