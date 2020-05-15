import cv2
print(cv2.__version__)

imagem = cv2.imread('pessoas/opencv-python.jpg')
imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)  # BGR Blue, Green and Red
cv2.imshow('Original', imagem)
cv2.imshow('Cinza', imagemCinza)
cv2.waitKey()
