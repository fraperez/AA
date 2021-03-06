import math

class Vector_Categorias():
    def __init__(self):
        self.male = [0, 0, "male", "male"]
        self.fem = [0, 0, "female", "female"]
        # catfem = [cantidad catfem, cantidad de vivos de fem,nombre,como buscarlo en el vector de busqueda]
        self.catsexo = [self.male, self.fem]

        self.clase1 = [0, 0, "clase 1", 1]
        self.clase2 = [0, 0, "clase 2", 2]
        self.clase3 = [0, 0, "clase 3", 3]
        self.catclase = [self.clase1, self.clase2, self.clase3]

        # dividimos las edades en 7 categorias
        # 1 < 10; 2 <20 3< 30 4<40 5<50 6 < 60 7 >60
        self.cat1 = [0, 0, "cat1","cat1"]
        self.cat2 = [0, 0, "cat2","cat2"]
        self.cat3 = [0, 0, "cat3","cat3"]
        self.cat4 = [0, 0, "cat4","cat4"]
        self.cat5 = [0, 0, "cat5","cat5"]
        self.cat6 = [0, 0, "cat6","cat6"]
        self.cat7 = [0, 0, "cat7","cat7"]


        self.cat_edad = [self.cat1, self.cat2, self.cat3, self.cat4,self.cat5,self.cat6,self.cat7]

        self.vector_categorias = [self.catsexo, self.catclase, self.cat_edad]

    def dar_categoria(self,numero):
        return(self.vector_categorias[numero])

    def dar_vector(self):
        return self.vector_categorias

class Vector_de_datos():
    def __init__(self,Datos_sex,Datos_clase,Datos_edad,Datos_Survived):
        self.datos_sex = Datos_sex
        i = 0
        self.datos_edad = []
        self.datos_clase = []
        while i < len (Datos_clase):
            self.datos_clase.append(categorizar_clase(Datos_clase[i]))
            self.datos_edad.append(categorizar_edad(Datos_edad[i]))
            i += 1
        self.datos_survived = Datos_Survived
        self.vector = [self.datos_sex,self.datos_clase,self.datos_edad,self.datos_survived]

    def dar_vector(self):
        return self.vector




class Nodo():
    def __init__(self,nombre,padre,tipo,generacion):
        print("Se creo el nodo hijo generacion ", generacion, "",nombre, "el padre es:", padre.dar_nombre() )
        self.nombre = nombre
        self.padre = padre
        self.generacion = generacion
        self.tipo = tipo
        self.hijo = []
        self.nuevo_vector_categorias = Vector_Categorias()
        self.actualizar_parametros()
        self.total,self.tot_vivo= self.contar_total()
        print()
        self.crear_nodo_hijo()


    def dar_nombre(self):
        return self.nombre

    def dar_tipo(self):
        return self.tipo

    def cantidad_hijos(self):
        return len(self.hijo)

    def dar_vector(self):
        return self.nuevo_vector_categorias

    def dar_categoria(self,num):
        return self.nuevo_vector_categorias.dar_categoria(num)

    def agregar_hijo(self,hijo1):
        self.hijo.append(hijo1)

    def dar_hijo(self,numero):
        return self.hijo[numero]

    def vector_hijo(self):
        return self.hijo

    def dar_padre(self):
        return self.padre

    #actualizo el vector de categorias, busco aquellos datos que cumplan la condicion de pertenecer a esta subcategoria
    def actualizar_parametros(self):
        if self.generacion == 1:
            i = 0
            while i < total-1:
                    if Datos.dar_vector()[self.tipo][i] == self.dar_nombre():
                        u = 0
                        while u<3:
                            if u != self.tipo:
                                actualizar_parametros_dado(u,Datos.dar_vector(),i,self.nuevo_vector_categorias.dar_vector())
                            u +=1
                    i += 1

        else:
            #si es de segunda generacion o superior tiene que que pretenecer a esta subclase y a la subclase del padre
            i = 0
            while i < total-1:

                if Datos.dar_vector()[self.tipo][i] == self.dar_nombre() and Datos.dar_vector()[self.padre.dar_tipo()][i]==self.padre.dar_nombre():
                    if(self.generacion ==2):
                        u = 0
                        while u < 3:
                            if u != self.tipo and u != self.padre.dar_tipo():
                                actualizar_parametros_dado(u, Datos.dar_vector(), i,self.nuevo_vector_categorias.dar_vector())
                            u += 1
                    #se es de tercera generacion tambien tiene que pertenecer a la subclase del padre del padre
                    else:
                        if Datos.dar_vector()[self.padre.dar_padre().dar_tipo()][i]==self.padre.dar_padre().dar_nombre():
                            actualizar_parametros_dado(self.tipo,Datos.dar_vector(),i,self.nuevo_vector_categorias.dar_vector())
                i += 1

    #una vez que actualizo los parametros puedo contar la cantidad total y la cantidad de vivos
    def contar_total(self):
        i = 0
        total = 0
        tot_vivo = 0
        if self.generacion != 3:
            while(i < 3):
                if i != self.tipo and i !=self.padre.dar_tipo():
                  for clases in self.nuevo_vector_categorias.dar_categoria(i):
                    total += clases[0]
                    tot_vivo += clases[1]
                  return  total,tot_vivo
                i += 1
        else:
            for clases in self.nuevo_vector_categorias.dar_categoria(self.dar_tipo()):
                total += clases[0]
                tot_vivo += clases[1]
            return total, tot_vivo

class Nodo_Shannon(Nodo):
    pass


    def Calcular_ganancias_Shannon (self,numero):
        h_vive = calcular_entropia(self.tot_vivo,self.total)
        h = dar_entropias(self.nuevo_vector_categorias.dar_categoria(numero))

        gain = calcular_ganancia_Shannon(h_vive,self.total,h,self.nuevo_vector_categorias.dar_categoria(numero))
        return gain

    def crear_nodo_hijo(self):
        if self.total == 0:
            muertos = total - total_vivo
            if total_vivo >= muertos:
                Sobrevive = Nodo_Terminal("Sobrevive")
                self.agregar_hijo(Sobrevive)
            else:
                No_Sobrevive = Nodo_Terminal("No Sobrevive")
                self.agregar_hijo(No_Sobrevive)
        elif self.total == self.tot_vivo:
            Sobrevive = Nodo_Terminal("Sobrevive")
            self.agregar_hijo(Sobrevive)

        elif self.tot_vivo == 0:
            No_Sobrevive = Nodo_Terminal("No Sobrevive")
            self.agregar_hijo(No_Sobrevive)
        else:
            i = 0
            Mayor = 0
            while i < 3:
                if i != self.tipo:
                    Mayor = mayor(Mayor, self.Calcular_ganancias_Shannon(i))
                i += 1
            p = 0
            while p < 3:
                if (p != self.tipo and Mayor == self.Calcular_ganancias_Shannon(p)):
                    while len(self.hijo) < len(vector_categorias.dar_categoria(p)):
                        v = self.nuevo_vector_categorias.dar_categoria(p)
                        Nodo_Hijo = Nodo_Shannon(v[len(self.hijo)][2], self, p, self.generacion + 1)
                        self.agregar_hijo(Nodo_Hijo)
                        print("")
                    break

                p += 1

class Nodo_Gini(Nodo):
    pass
    def Calcular_ganancias_Gini(self,numero):
        i = 0
        gain = 1
        while i<len(self.nuevo_vector_categorias.dar_categoria(numero)):
            gain -= dar_cantidad_vivos(self.nuevo_vector_categorias.dar_categoria(numero),i)/total
            i += 1
        return gain

    def crear_nodo_hijo(self):
        if self.total == 0:
            muertos = total - total_vivo
            if total_vivo >= muertos:
                Sobrevive = Nodo_Terminal("Sobrevive")
                self.agregar_hijo(Sobrevive)
            else:
                No_Sobrevive = Nodo_Terminal("No Sobrevive")
                self.agregar_hijo(No_Sobrevive)

        elif self.total == self.tot_vivo:
            Sobrevive = Nodo_Terminal("Sobrevive")
            self.agregar_hijo(Sobrevive)

        elif self.tot_vivo == 0:
            No_Sobrevive = Nodo_Terminal("No Sobrevive")
            self.agregar_hijo(No_Sobrevive)
        else:
            i = 0
            Mayor = 0
            while i < 3:
                if i != self.tipo and i!=self.padre.dar_tipo():
                    Mayor = mayor(Mayor, self.Calcular_ganancias_Gini(i))
                i += 1
            p = 0
            while p < 3:

                if (p != self.tipo and p!= self.padre.dar_tipo() and Mayor == self.Calcular_ganancias_Gini(p)):
                    while len(self.hijo) < len(vector_categorias.dar_categoria(p)):
                        v = self.nuevo_vector_categorias.dar_categoria(p)
                        Nodo_Hijo = Nodo_Gini(v[len(self.hijo)][2], self, p, self.generacion + 1)
                        self.agregar_hijo(Nodo_Hijo)
                        print("")
                    break

                p += 1




class Nodo_Raiz():

    def __init__(self,nombre, vector_de_la_cat,tipo):
        print("Se creo el nodo raiz", nombre, "del tipo ", tipo)
        self.nombre = nombre
        self.hijo = []
        self.vector_de_la_cat = vector_de_la_cat
        self.tipo = tipo
        for subcategoria in self.vector_de_la_cat:
            self.agregar_hijo(subcategoria)

    def vector_hijo(self):
        return self.hijo

    def dar_hijo(self,numero):
        return self.hijo[numero]

    def cantidad_hijos(self):
        return len(self.hijo)

    def dar_tipo(self):
        return self.tipo
    def dar_nombre(self):
        return self.nombre

class Nodo_Raiz_Shannon(Nodo_Raiz):
    pass

    def agregar_hijo(self, subcategoria):
        hijo = Nodo_Shannon(subcategoria[2], self, self.tipo, 1)
        self.hijo.append(hijo)

class Nodo_Raiz_Gini(Nodo_Raiz):
    pass

    def agregar_hijo(self, subcategoria):
        hijo = Nodo_Gini(subcategoria[2], self, self.tipo, 1)
        self.hijo.append(hijo)




class Nodo_Terminal():
    def __init__(self,nombre):
        print("Se creo el nodo terminal", nombre)
        self.nombre = nombre

    def dar_nombre(self):
        return self.nombre

    def cantidad_hijos(self):
        return 0






def categorizar_edad(Dato_edad):
        i = 0
        if (Dato_edad > 70):
            return "cat7"

        while (i < 7):
            if (Dato_edad < 10 * i):
                return("cat" + str(i))
            i += 1

def categorizar_clase(Dato_clase):
        i = 1
        while i<4:
            if Dato_clase == i:
                return ("clase "+str(i))
            i += 1




def contabilizar_sexo(dato_sexo,numero,catsexo):
    if (dato_sexo == "male"):
        catsexo[0][0] += 1
        if(Survived[numero] == 1):
            catsexo[0][1] += 1
    else:
        catsexo[1][0] += 1
        if (Survived[numero] == 1):
            catsexo[1][1] += 1

def contabilizar_edad(dato_edad,numero,catedad):
    i = 1

    while (i < 7):
        if (dato_edad == "cat" + str(i)):
                catedad[i-1][0] += 1

                if(Survived[numero] == 1):
                    catedad[i-1][1] += 1
        i += 1

def contabilizar_clase(dato_clase,numero,catclase):
    i = 1
    while i<4:
        if(dato_clase == "clase " + str(i)):
             catclase[i-1][0] += 1
             if(Survived[numero] == 1):
               catclase[i-1][1] += 1
        i += 1

def actualizar_parametros_dado(tipo,vector_datos,numero, vector_todas_las_categorias):
    if tipo == 0:
        contabilizar_sexo(vector_datos[0][numero],numero,vector_todas_las_categorias[0])
    elif tipo == 1:
        contabilizar_clase(vector_datos[1][numero],numero,vector_todas_las_categorias[1])
    else:
        contabilizar_edad(vector_datos[2][numero],numero,vector_todas_las_categorias[2])

def contabilizar_todo(vector_datos,vector_de_todas_las_cat):
    i = 0
    while i < total:
        contabilizar_sexo(vector_datos[0][i],i,vector_de_todas_las_cat[0])
        contabilizar_clase(vector_datos[1][i],i,vector_de_todas_las_cat[1])
        contabilizar_edad(vector_datos[2][i],i,vector_de_todas_las_cat[2])
        i += 1





def dar_cantidad_vivos(categoria,numero):
    return categoria[numero][1]

def dar_cantidad(categoria,numero):
    return categoria[numero][0]





def calcular_entropia(positivos,total):
    negativos = total - positivos
    if negativos == 0:
        if positivos == 0:
            h=0
        else:
            h = -(positivos / total) * math.log(positivos / total, 2)
    elif positivos == 0:
        h = - (negativos / total) * math.log(negativos / total, 2)
    else:
        h = -(positivos/total)*math.log(positivos/total,2) - (negativos/total)*math.log(negativos/total,2)
    return h

def dar_entropias(cat):
    i = 0
    h = []
    while i < len(cat):
        h.append(calcular_entropia(dar_cantidad_vivos(cat, i), dar_cantidad(cat, i)))
        i += 1
    return h

def calcular_ganancia_Shannon(h_vive,total,h,cat):
    i = 0
    gain = h_vive
    while i < len(h):
        gain -= (dar_cantidad(cat, i) / total) * h[i]
        i += 1
    return gain



def calcular_ganancia_Gini(total,cat):
    i = 0
    gain = 1
    while i<len(cat):
        gain -= dar_cantidad_vivos(cat,i)/total
        i += 1
    return gain




def total_vida(Survived):
    i = 0
    tot_vive = 0
    for vida in Survived:
        if (vida == 1):
            tot_vive += 1
        i += 1
    return tot_vive

def mayor(a,b):
    if a>=b:
        return a
    else:
        return b





def dar_Nodo_raiz_Shannon(vector_categorias,total):
    tot_vive = total_vida(Survived)

    h_vive = calcular_entropia(tot_vive, total)

    h_sex = dar_entropias(vector_categorias[0])

    h_clase = dar_entropias(vector_categorias[1])

    h_edad = dar_entropias(vector_categorias[2])

    gain_sex = calcular_ganancia_Shannon(h_vive, total, h_sex, vector_categorias[0])

    gain_clase = calcular_ganancia_Shannon(h_vive, total, h_clase, vector_categorias[1])

    gain_edad = calcular_ganancia_Shannon(h_vive, total, h_edad, vector_categorias[2])

    m1 = mayor(gain_sex, gain_clase)
    Mayor = mayor(m1, gain_edad)

    if Mayor == gain_sex:
         return Nodo_Raiz_Shannon("sexo",vector_categorias[0],0)
    elif Mayor == gain_clase:
        return Nodo_Raiz_Shannon("clase",vector_categorias[1],1)
    else:
        return Nodo_Raiz_Shannon("edad",vector_categorias[2],2)


def dar_Nodo_raiz_Gini(vector_categorias,Survived):
    total = len(Survived)

    gain_sex = calcular_ganancia_Gini(total, vector_categorias[0])

    gain_clase = calcular_ganancia_Gini(total, vector_categorias[1])

    gain_edad = calcular_ganancia_Gini(total, vector_categorias[2])

    m1 = mayor(gain_sex, gain_clase)
    Mayor = mayor(m1, gain_edad)

    if Mayor == gain_sex:
         return Nodo_Raiz_Gini("sexo",vector_categorias[0],0)
    elif Mayor == gain_clase:
        return Nodo_Raiz_Gini("clase",vector_categorias[1],1)
    else:
        return Nodo_Raiz_Gini("edad",vector_categorias[2],2)

def sobrevivio(datos,Nodo_Raiz):
    datos[1] = categorizar_clase(datos[1])
    datos[2] = categorizar_edad(datos[2])
    i = 0
    while i<3:
        if i == Nodo_Raiz.dar_tipo():
            for hijo in Nodo_Raiz.vector_hijo():
                if datos[hijo.dar_tipo()] == hijo.dar_nombre():
                    if len(hijo.vector_hijo()) == 1:
                        return sob(hijo.dar_hijo(0).dar_nombre())
                    else:
                        for hijo2 in hijo.vector_hijo():
                            if datos[hijo2.dar_tipo()] == hijo2.dar_nombre():
                                if len(hijo2.vector_hijo()) == 1:
                                    return sob(hijo2.dar_hijo(0).dar_nombre())
                                else:
                                    for hijo3 in hijo2.vector_hijo():
                                        sob(hijo3.dar_hijo(0).dar_nombre())
        i +=1


def sob(nombre):
    if nombre == "Sobrevive":
        return "Sobrevive"
    elif nombre == "No Sobrevive":
        return "No Sobrevive"



sexo = ["female","female","female","male","male","male","male","male","female","male"]
age = [52,71,74,7,2,25,12,5,12,5]
Pclass = [3,2,3,1,2,2,2,3,1,1]
Survived = [0,1,0,0,0,0,1,1,1,0]
total = len(Survived)
total_vivo = total_vida(Survived)
Nombre_categorias = ["sexo","Clase","Edad"]

Datos = Vector_de_datos(sexo,Pclass,age,Survived)



vector_categorias = Vector_Categorias()


#tipo 0 sexo
#tipo 1 clase
#tipo 2 edad


contabilizar_todo(Datos.dar_vector(),vector_categorias.dar_vector())
print("Esto es Shannon")
print("")
NodoRaiz = dar_Nodo_raiz_Shannon(vector_categorias.dar_vector(),total)
print("")
print("")
print("esto es de Gini")
print("")
print("")

NodoRaiz2 = dar_Nodo_raiz_Gini(vector_categorias.dar_vector(),Survived)

i = 0
print("termino")

comparar = ["male",2,85,1]

print(sobrevivio(comparar,NodoRaiz))
comparar = ["male",2,85,1]

print(sobrevivio(comparar,NodoRaiz2))
