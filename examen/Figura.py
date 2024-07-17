import os
import cv2
import math
import face_recognition


class Figura:
    def __init__(self, ruta_empleados):
        if not os.path.exists(ruta_empleados):
            raise FileNotFoundError(f"La ruta {ruta_empleados} no existe.")
        self.ruta = ruta_empleados
        self.mis_imagenes = []
        self.nombres_empleados = []
        self.lista_empleados = os.listdir(self.ruta)

        for empleado in self.lista_empleados:
            imagen_actual = cv2.imread(f"{self.ruta}/{empleado}")
            if imagen_actual is not None:
                self.mis_imagenes.append(imagen_actual)
                self.nombres_empleados.append(os.path.splitext(empleado)[0])

        self.lista_empleados_codificada = self.codificar(self.mis_imagenes)
        self.figura_usuario = None
        self.radio = float(input("Determine el valor de R: "))

    def codificar(self, imagenes):
        imagenes_codificadas = []
        for imagen in imagenes:
            if imagen is not None:
                imagen_codificada = face_recognition.face_encodings(imagen)
                if imagen_codificada:
                    imagenes_codificadas.append(imagen_codificada[0])
        return imagenes_codificadas

    def reconocer_rostro(self, imagen_actual):
        if imagen_actual is None:
            raise ValueError("La imagen actual no se pudo cargar. Verifica la ruta de la imagen.")

        rgb_imagen = cv2.cvtColor(imagen_actual, cv2.COLOR_BGR2RGB)
        codificaciones_actuales = face_recognition.face_encodings(rgb_imagen)

        if codificaciones_actuales:
            coincidencias = face_recognition.compare_faces(self.lista_empleados_codificada, codificaciones_actuales[0])
            return True if True in coincidencias else False
        return False

    def calcular(self):
        if self.figura_usuario == "circulo":
            area = math.pi * self.radio ** 2
            print(f"El área del círculo es: {area}")
        elif self.figura_usuario == "esfera":
            volumen = (4 / 3) * math.pi * self.radio ** 3
            print(f"El volumen de la esfera es: {volumen}")

    def set_figura(self, tipo_figura):
        self.figura_usuario = tipo_figura



from ClaseRF import ReconocimientoFacial

ruta_empleados = r"C:\Users\hogar\PycharmProjects\FaceRecognition\.venv\Empleados"
figura_usuario = input("Ingrese la figura que quiere ver ").lower()

try:
    figura = Figura(ruta_empleados)
    figura.set_figura(figura_usuario)

    # Inicializa el reconocimiento facial con la ruta de las imágenes de los empleados
    reconocimiento_facial = ReconocimientoFacial(ruta_empleados)

    # Captura una imagen actual
    imagen_actual = reconocimiento_facial.capturar_imagen()

    if imagen_actual is None:
        raise ValueError("No se pudo capturar la imagen actual.")

    rostro_reconocido = figura.reconocer_rostro(imagen_actual)

    if rostro_reconocido:
        figura.calcular()
    else:
        figura.set_figura("esfera" if figura_usuario == "circulo" else "circulo")
        figura.calcular()
except FileNotFoundError as e:
    print(e)
except ValueError as e:
    print(e)
