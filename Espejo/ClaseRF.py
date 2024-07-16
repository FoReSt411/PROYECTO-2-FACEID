import cv2
import os
import face_recognition as fr
import numpy as np
from datetime import datetime

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
                                  200
                                  )
                    # Mostrar fecha y hora en la imagen
                    time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    cv2.putText(imagen, time, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(imagen, self.nombres_empleados[indice_coincidencia], (caraubic[3], caraubic[0] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.imshow("Empleado Reconocido", imagen)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()

                    print(f"Bienvenido {self.nombres_empleados[indice_coincidencia]}")
                    #mostrar imagen empleado y foto capturada
                    # cv2.imshow("Empleado", self.mis_imagenes[indice_coincidencia])
                    # cv2.imshow("Foto Empleado", imagen)
                    cv2.imshow("Empleado Reconocido", imagen)
                    return True
        print("No se encontraron coincidencias")
        return False