import cv2
import os
import face_recognition as fr
import numpy as np
import math
from datetime import datetime

class Figura:
    def __init__(self, tipo, radio=None, lado=None, base=None, altura=None):
        self.tipo = tipo.lower()
        self.radio = radio
        self.lado = lado
        self.base = base
        self.altura = altura

    def area(self):
        if self.tipo == 'circulo' and self.radio:
            return math.pi * self.radio ** 2
        elif self.tipo == 'esfera' and self.radio:
            return 4 * math.pi * self.radio ** 2
        elif self.tipo == 'cuadrado' and self.lado:
            return self.lado ** 2
        elif self.tipo == 'rectangulo' and self.base and self.altura:
            return self.base * self.altura
        else:
            return None

    def volumen(self):
        if self.tipo == 'esfera' and self.radio:
            return (4 / 3) * math.pi * self.radio ** 3
        elif self.tipo == 'cubo' and self.lado:
            return self.lado ** 3
        else:
            return None

class ReconocimientoFacial:
    def __init__(self, ruta_empleados):
        self.ruta = ruta_empleados
        self.mis_imagenes = []
        self.nombres_empleados = []
        self.lista_empleados = os.listdir(self.ruta)
        for empleado in self.lista_empleados:
            imagen_actual = cv2.imread(f"{self.ruta}/{empleado}")
            self.mis_imagenes.append(imagen_actual)
            self.nombres_empleados.append(os.path.splitext(empleado)[0])
        self.lista_empleados_codificada = self.codificar(self.mis_imagenes)

    def codificar(self, imagenes):
        lista_codificada = []
        for imagen in imagenes:
            imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
            codificados = fr.face_encodings(imagen)
            if codificados:
                lista_codificada.append(codificados[0])
        return lista_codificada

    def capturar_imagen(self):
        captura = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        exito, imagen = captura.read()
        captura.release()
        if exito:
            cv2.imshow("Foto Empleado", imagen)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            return imagen
        else:
            print("No se pudo tomar la foto")
            return None

    def reconocer_empleado(self, imagen):
        if imagen is None:
            return False
        cara_captura = fr.face_locations(imagen)
        cara_captura_codificada = fr.face_encodings(imagen, known_face_locations=cara_captura)
        for caracodif, caraubic in zip(cara_captura_codificada, cara_captura):
            coincidencias = fr.compare_faces(self.lista_empleados_codificada, caracodif, 0.6)
            distancias = fr.face_distance(self.lista_empleados_codificada, caracodif)
            if True in coincidencias:
                indice_coincidencia = np.argmin(distancias)
                if distancias[indice_coincidencia] <= 0.6:
                    # Mostrar imagen y nombre del empleado
                    cv2.rectangle(imagen,
                                  (caraubic[3], caraubic[0]),
                                  (caraubic[1], caraubic[2]),
                                  (0, 255, 0),
                                  2
                                  )
                    # Mostrar fecha y hora en la imagen
                    time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    cv2.putText(imagen, time, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(imagen, self.nombres_empleados[indice_coincidencia], (caraubic[3], caraubic[0] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    # Redimensionar imágenes para que tengan la misma altura y mantener la relación de aspecto
                    height = min(imagen.shape[0], self.mis_imagenes[indice_coincidencia].shape[0])

                    # Calcular la relación de aspecto para mantenerla
                    aspect_ratio_empleado = self.mis_imagenes[indice_coincidencia].shape[1] / \
                                            self.mis_imagenes[indice_coincidencia].shape[0]
                    new_width_empleado = int(height * aspect_ratio_empleado)
                    imagen_empleado_resized = cv2.resize(self.mis_imagenes[indice_coincidencia],
                                                         (new_width_empleado, height))

                    aspect_ratio_capturada = imagen.shape[1] / imagen.shape[0]
                    new_width_capturada = int(height * aspect_ratio_capturada)
                    imagen_resized = cv2.resize(imagen, (new_width_capturada, height))

                    # Combinar la imagen capturada y la imagen almacenada
                    imagen_combinada = np.hstack((imagen_empleado_resized, imagen_resized))

                    cv2.imshow("Empleado Reconocido", imagen_combinada)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()

                    print(f"Bienvenido {self.nombres_empleados[indice_coincidencia]}")
                    return True
        print("No se encontraron coincidencias")
        return False


# Ejemplo de uso
if __name__ == "__main__":
    reconocimiento_facial = ReconocimientoFacial(Empleados)
    imagen_capturada = reconocimiento_facial.capturar_imagen()
    reconocido = reconocimiento_facial.reconocer_empleado(imagen_capturada)

    # Crear instancia de Figura y calcular área o volumen según el reconocimiento
    if reconocido:
        figura = Figura(tipo='circulo', radio=5)  # Ejemplo con un círculo de radio 5
        print(f"Área de la figura: {figura.area()}")
    else:
        figura = Figura(tipo='esfera', radio=5)  # Ejemplo con una esfera de radio 5
        print(f"Volumen de la figura: {figura.volumen()}")
