# TP 2 Machine Learning ITBA

## Intrucciones de uso:
### Generear ambiente de trabajo (virtual env)
- ` pip3 install virtualenv`
- `cd AA`
- `virtualenv venv -p Python3`
- `source venv/bin/activate` activa el virtual envirounment ( en la consola te tiene que aparece algo como `(venv) ...`)
- `pip3 install -r requirements.txt` esto instala todas las dependencias que usa el proyecto
- ahora estamos en condiciones de correr el proyecto

DISCLAIMNER: TODOS LOS PASOS ASUMEN QUE SE ENCUENTRA DENTRO DEL VIRTUAL ENVIROUMENT
* Decision Trees:
  -  `python titanic_survival_prediction/console.py [PARAMS]`
  - parametros a utilizar:
    * `-d`: path al alrchivo dcsv
    * `-t`: porcentage de data para el test
    * `-p`: predictor a usar (puede ser `decsion_tree` o `random_forest`)
    * `-g`: función de ganancia a utilizar (puede ser `shanon` o `gini`)
    * `-ndata`: en caso de utiliar `random_forest` hay que especificar el porcentage de data para cada árbol en particular
    * `-nattr`: en caso de utiliar `random_forest` hay que especificar el número de atributos a considerar por cada árbol
    * `-ntree`: en caso de utiliar `random_forest` hay que especificar el número árboles a usar


* KNN:
  - `python sentiment_analysis/console.py [PARAMS]`
  - parametros a utilizar:
    * `-d`: path al alrchivo dcsv
    * `-t`: porcentage de data para el test
    * `-k`: numero de vecinos a utilizar (debe ser un número impar)
    * `-w`: se especifica si usar pesos o no (no tiene parametros)

    