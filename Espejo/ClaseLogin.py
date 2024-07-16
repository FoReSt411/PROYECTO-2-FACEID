import bcrypt


class Login:
    def __init__(self, usuario_correcto, intentos, reconocimiento_facial):
        self.usuario_correcto = usuario_correcto
        self.intentos = intentos
        self.reconocimiento_facial = reconocimiento_facial

    def autenticar(self):
        while self.intentos > 0:
            metodo = input("Seleccione el método de autentificación (1: Contraseña, 2: Reconocimiento Facial): ")

            if metodo == '1':
                if self.autenticar_contraseña():
                    return True
            elif metodo == '2':
                if self.autenticar_reconocimiento_facial():
                    return True
            else:
                print("Opción no válida.")

        return False

    def autenticar_contraseña(self):
        pwd = bytes('pololo', 'utf-8')
        sal = bcrypt.gensalt()
        encriptado = bcrypt.hashpw(pwd, sal)

        while self.intentos > 0:
            usuario = input("Ingrese su usuario: ")
            contraseña = bytes(input("Ingrese su contraseña: "), 'utf-8')

            if usuario == self.usuario_correcto and bcrypt.checkpw(contraseña, encriptado):
                print("Acceso concedido")
                return True
            else:
                self.intentos -= 1
                if self.intentos > 0:
                    print(f"Usuario o contraseña incorrecta, intente de nuevo ({self.intentos} restantes)")
                    input("Presione Enter para intentarlo de nuevo...")

        return False

    def autenticar_reconocimiento_facial(self):
        while self.intentos > 0:
            imagen_capturada = self.reconocimiento_facial.capturar_imagen()
            if self.reconocimiento_facial.reconocer_empleado(imagen_capturada):
                print("Acceso concedido")
                return True
            else:
                self.intentos -= 1
                if self.intentos > 0:
                    print(f"Reconocimiento facial fallido, intente de nuevo ({self.intentos} restantes)")
                    input("Presione Enter para intentarlo de nuevo...")

        return False