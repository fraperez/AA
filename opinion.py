import math
#posiciones
# 0 Rewiev_Title
# 1 wordcount
# 2 titleSentiment
# 3 textSentiment
# 4 Star_Rating
# 5 sentimentValue
class Vector_de_datos():
    def __init__(self,Review_Title,wordcount,titleSentiment,textSentiment,Star_Rating,sentimentValue):
        self.Review_Title = Review_Title
        self.wordcount = wordcount
        self.Star_Rating = Star_Rating
        self.sentimentValue = sentimentValue
        i = 0
        self.titleSentiment = []
        self.textSentiment = []
        while i < len (titleSentiment):
            self.titleSentiment.append(categorizar_calificacion(titleSentiment[i]))
            self.textSentiment.append(categorizar_calificacion(textSentiment[i]))
            i += 1
        self.vector = [self.Review_Title,self.wordcount,self.titleSentiment,self.textSentiment,self.Star_Rating,self.sentimentValue]

    def dar_vector(self):
        return self.vector

    def dar_categoria(self,numero):
        return self.vector[numero]

    def categorizar_calificacion(calificacion):
        if calificacion == "positive":
            return 1
        elif calificacion == "negative":
            return 0
        # este es el caso de N/A
        else:
            return 0.5


def categorizar_calificacion(calificacion):
        if calificacion == "positive":
            return 1
        elif calificacion == "negative":
            return 0
        # este es el caso de N/A
        else:
            return 0.5


def dar_Star_Rating(k,Vector_de_datos,Vector_a_analizar):
    i = 0
    categorizar_vector(Vector_a_analizar)
    while i<len(Vector_de_datos[0]):
        distancia = dar_distancia(Vector_de_datos, i ,Vector_a_analizar)
        star_Rating = Vector_de_datos[4][i]
        posicion = i
        vector_individual = [distancia,star_Rating,posicion]
        if i<k:
            if i == 0:
                vector_total = vector_individual
            else:
                vector_total = ordenar_y_agregar(vector_total,vector_individual,i)
        elif distancia<vector_total[0][0]:
            vector_total = ordenar(vector_total,vector_individual)
        i +=1
    contar_Star_Rating = [0,0,0,0,0]
    for vector_individual in vector_total:
        rating = vector_individual[1]
        i = 1
        while i < 5:
            if rating == i:
                contar_Star_Rating[i-1] += 1
                break
            i +=1

    rating_final = posicion_mayor(contar_Star_Rating) + 1
    print ("el rating de este comentario es: ",rating_final)

def categorizar_vector(vector):
    vector [2] = categorizar_calificacion(vector[2])
    vector [3] = categorizar_calificacion(vector[3])

def dar_Star_Rating_ponderada(k,Vector_de_datos,Vector_a_analizar):
    i = 0
    categorizar_vector(Vector_a_analizar)
    while i<len(Vector_de_datos[0]):
        distancia = dar_distancia(Vector_de_datos, i ,Vector_a_analizar)
        star_Rating = Vector_de_datos[4][i]
        posicion = i
        vector_individual = [distancia, star_Rating, posicion]
        if i<k:
            if i == 0:
                vector_total = vector_individual
            else:
                vector_total = ordenar_y_agregar(vector_total,vector_individual,i)
        elif distancia<vector_total[0][0]:
            vector_total = ordenar(vector_total,vector_individual)
        i +=1
    contar_Star_Rating_Ponderada = [0,0,0,0,0]
    
    
    #aca cambia con respecto a no ponderada
    dist_0 = 0
    for vector_individual in vector_total:
        rating = vector_individual[1]
        i = 1
        while i < 5:
            if rating == i:
                distancia = dar_distancia(Vector_de_datos,vector_individual[2],Vector_a_analizar)
                if distancia == 0:
                    dist_0 = 1
                    print("el rating de este comentario es: ", rating)
                else:
                    contar_Star_Rating_Ponderada[i-1] += 1/dar_distancia(Vector_de_datos,vector_individual[2],Vector_a_analizar)
                    break
            i +=1

    if dist_0 == 0:
        rating_final = posicion_mayor(contar_Star_Rating_Ponderada) + 1
        print ("el rating de este comentario es: ",rating_final)






def promedio_palabras_StarRating_1(Categoria_StarRating,Categoria_wordcount):
    total_palabras = 0
    total_rating_1 = 0
    i = 0
    while i < len(Categoria_StarRating):
        if Categoria_StarRating[i] == 1:
            total_palabras += Categoria_wordcount[i]
            total_rating_1 += 1
           i += 1     
    promedio = total_palabras/total_rating_1
    print("El promedio de palabras utilizadas en comentarios de rating 1 es: ",promedio)

def dar_distancia(Vector_dato,posicion, Vector_a_analizar):
    arg_titleSentiment = (Vector_dato[2][posicion]-Vector_a_analizar[2])**2
    arg_textSentiment = (Vector_dato[3][posicion]-Vector_a_analizar[3])**2
    arg_textValue = (Vector_dato[5][posicion]-Vector_a_analizar[5])**2
    distancia = math.sqrt(arg_textSentiment + arg_titleSentiment + arg_textValue)

    return distancia



def posicion_mayor(vector):
    #numero_mayor = [numero,posicion]
    numero_mayor = [0,0]
    i = 0
    while i < len(vector):
        numero_mayor[0] = mayor(vector[i],numero_mayor[0])
        if vector[i] == numero_mayor[0]:
            numero_mayor[1] = i
        i += 1
    return numero_mayor[1]

def mayor(a,b):
    if a>b:
        return a
    else:
        return b

def ordenar(vector_total,vector_individual):
    posicion = 0
    nuevo_vector = []
    agregado = 0;
    for vector in vector_total:
        if vector_individual[0] < vector[0]:
            #quiero sacar el primer vector
            if posicion >= 1:
                nuevo_vector.append(vector)
            posicion += 1
        else:
            if agregado == 0:
                agregado = 1
                nuevo_vector.append(vector_individual)
                nuevo_vector.append(vector)
            else:
                nuevo_vector.append(vector)
    if agregado == 0:
        nuevo_vector.append(vector_individual)
    return nuevo_vector

def ordenar_y_agregar(vector_total,vector_individual,largo):
    nuevo_vector = []
    agregado = 0;
    if largo == 1:
        if vector_individual[0] < vector_total[0]:
            nuevo_vector.append(vector_total)
            nuevo_vector.append(vector_individual)
        else:
            nuevo_vector.append(vector_individual)
            nuevo_vector.append(vector_total)
    else:
        for vector in vector_total:
            if agregado == 0:
                if vector_individual[0] < vector[0]:
                    nuevo_vector.append(vector)
                else:
                    agregado = 1
                    nuevo_vector.append(vector_individual)
                    nuevo_vector.append(vector)
            else:
                nuevo_vector.append(vector)
        if agregado == 0:
            nuevo_vector.append(vector_individual)

    return nuevo_vector


ReviewTitle = ["a","a","a","a","a","a","a"]
wordcount = [53,43,65,78,76,22,12]
titlesentiment = ["positive", "positive"," ", "positive", "negative","negative","negative"]
textsentiment = titlesentiment
Star_Rating = [1,1,2,2,2,3,3]
sentimental_value = Star_Rating

vector = Vector_de_datos(ReviewTitle,wordcount,titlesentiment,textsentiment,Star_Rating,sentimental_value)
vec_analizar = ["a",1,"negative","negative",3,2.5]



dar_Star_Rating(3,vector.dar_vector(),vec_analizar)

dar_Star_Rating_ponderada(3,vector.dar_vector(),vec_analizar)
