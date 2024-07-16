from Departamentos import departamentos_municipios
class DNI:
    def __init__(self):
        self.dni = input("Ingrese su número de identidad: ")

    def validar_dni(self):
        if not self.dni.isdigit() or len(self.dni) != 13:
            print("Número de identidad inválido, intente de nuevo (13 dígitos y solo enteros)")
            input("Presione Enter para continuar...")
        else:
            self.procesar_dni()

    def procesar_dni(self):
        id_departamento = self.dni[0:2]
        id_municipio = self.dni[2:4]
        id_anionacimiento = self.dni[4:8]
        edad = 2024 - int(id_anionacimiento)

        if id_departamento not in departamentos_municipios or id_municipio not in departamentos_municipios[id_departamento]['municipios']:
            print("Datos no encontrados.")
            input("Presione Enter para continuar...")
        else:
            departamento = departamentos_municipios[id_departamento]
            municipio = departamento['municipios'].get(id_municipio)
            print("Departamento: ", departamento['nombre'])
            print("Municipio: ", municipio)
            print("Edad: ", edad, " años")

            if edad >= 21:
                print("Eres mayor de edad")
            elif edad >= 18:
                print("Eres ciudadano")
            else:
                print("Eres menor de edad")

            input("Presione Enter para continuar...")