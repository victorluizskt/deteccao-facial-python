import cv2

# Parâmetro 0, é para abrir qual camêra deseja para detecção de face.
video = cv2.VideoCapture(0)

# Criando a deteccao facial
classificadorFace = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')


while True:
    # Conectado é só pra saber se a imagem está sendo captada.
    conectado, frame = video.read()

    # Fazendo a deteccção de face
    frameCinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    facesDetectadas = classificadorFace.detectMultiScale(frameCinza, minSize=(60, 60))

    for (x, y, l, a) in facesDetectadas:
        cv2.rectangle(frame, (x, y), (x + l, y + a), (0, 0, 255), 2)

    # Frame vai conter o video ao vivo do programa.
    cv2.imshow('Video', frame)

    # ord faz a conversão da tabela asc para sair do programa.
    if cv2.waitKey(1) == ord('q'):
        break

# Release é para tirar da memória a WebCam para não gastar muita memória.
video.release()
cv2.destroyAllWindows()

