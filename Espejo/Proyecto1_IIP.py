from ClaseLogin import Login
from InspeccionarDept import InspeccionarDept
from ClaseDNI import DNI
from datetime import datetime
from ClaseRF import ReconocimientoFacial
'''
    ALUMNOS:
    - ILIANA LICETH ZUNIGA ENAMORADO
    - DIANY LIZBETH ENAMORADO FERNANDEZ
    - ANDERSON JAIR GARCIA MENJIVAR
    - TOMY JOSE MONTUFAR ZUNIGA
    - ANROLD STANLY FORD MADRID
'''
def main():


    usuario_correcto = "Ford"
    intentos = 10
    reconocimiento_facial = ReconocimientoFacial("Empleados")

    # Acceder al programa
    login = Login(usuario_correcto, intentos, reconocimiento_facial)
    while True:
        if login.autenticar():
            break
        elif login.intentos == 0:
            print("Ha excedido el número de intentos permitidos.")
            exit()

    # Menu principal
    while True:
        print("Bienvenido al programa")
        print("1. Inspeccionar departamentos y municipios.")
        print("2. Ingresar su cedula.")
        print("3. Salir.")
        selec = input("Elija una opcion: ")

        # Inspeccionar departamentos y municipios
        if selec == '1':
            inspeccion = InspeccionarDept()
        # Ingresar su cedula
        elif selec == '2':
            dni = DNI()
            dni.validar_dni()

        # Salir
        elif selec == '3':
            print("Programa finalizado.")
            exit()
        else:
            print("Opcion no valida, intente de nuevo.")
            input("Presione Enter para continuar...")

if __name__ == "__main__":
    main()