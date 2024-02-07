**PROYECTO INDIVIDUAL Nº1**
**Machine Learning Operations**

------------

**Presentado por **
***Kensit Marian Cortés Nemogá***

------------
**Acerca del proyecto**
<div style="text-align: justify;">
Steam, una plataforma líder en juegos para PC, necesita mejorar su sistema de recomendación. Este proyecto busca crear un sistema más eficaz y personalizado usando análisis de datos y aprendizaje automático. El propósito es ofrecer sugerencias de juegos más pertinentes para mejorar la satisfacción del usuario, impulsar el compromiso y aumentar las ventas y los ingresos de la plataforma.

------------


**Herramientas utilizadas**
-PYTHON 
-PANDAS
-FASTAPI
-RENDER

------------

**Estrategia de trabajo **

------------


**Extraccion, Transformacion y Carga de datos**

**Ingesta de datos**
Para iniciar nuestro proceso ETL, contaremos con tres carpetas de archivos:
		- Steam_games.json
		- User_reviews.json 
		- Users_items_json
 
De los cuales obtendremos los siguientes archivos respectivamente: 
- Output_steam_games
- Australian_user_reviews
- Australian_users_items
**Transformacion de datos**

Se necesitan de las siguientes librerias para abrir y leer los archivos: 

```python
import os
import json
import pandas as pd
import os.path
```
Para abrir el archivo output_steam_games, lo haacemos de la siguiente manera: 
```python
# Obtén la ruta al directorio 'models'
models_directory = os.path.join(os.path.dirname('__file__'), '../models')

def cargar_json_y_convertir_a_dataframe(nombre_archivo):
    ruta_archivo = os.path.join(models_directory, nombre_archivo)

    try:
        dataframe = pd.read_json(ruta_archivo, lines=True)
        return dataframe
    except FileNotFoundError:
        # Manejar la situación en la que el archivo no existe
        return None
    except json.JSONDecodeError as e:
        # Manejar errores de decodificación JSON
        print(f"Error al decodificar el archivo JSON: {e}")
        return None
```
Como ejemplo de uso tenemos: 
```python
archivo1_dataframe =cargar_json_y_convertir_a_dataframe('output_steam_games.json')
if archivo1_dataframe is not None:
		print(archivo1_dataframe.head(5))
```
Resultado
```python
  publisher genres app_name title   url release_date  tags reviews_url specs   
0      None   None     None  None  None         None  None        None  None  \
1      None   None     None  None  None         None  None        None  None   
2      None   None     None  None  None         None  None        None  None   
3      None   None     None  None  None         None  None        None  None   
4      None   None     None  None  None         None  None        None  None   

  price  early_access  id developer  
0  None           NaN NaN      None  
1  None           NaN NaN      None  
2  None           NaN NaN      None  
3  None           NaN NaN      None  
4  None           NaN NaN      None  
```
Eliminamos los valores nulos
```python
archivo1_dataframe = archivo1_dataframe.dropna(how='any')
```
Eliminamos las columnas que no se van a utilizar 
```python
columnas=['app_name','url','reviews_url','specs','early_access']
archivo1_dataframe = archivo1_dataframe.drop(columnas, axis=1)

```
Resultado

```python
archivo1_dataframe.columns.tolist()
```
['publisher', 'genres', 'title', 'release_date', 'tags', 'price', 'developer','id']

Verificamos la columna "release_date"para verificar que este ben el formato, ya que necesitamos extraer el año 
```python
archivo1_dataframe.release_date.sort_values()
```
Resultado
```python
89687     1983-06-19
89855     1984-04-29
110028    1984-11-01
116377    1985-01-01
99223     1986-05-01
             ...
109456      Oct 2016
101344          SOON
120318         SOON™
119928      Sep 2009
90917       Sep 2014
Name: release_date, Length: 22530, dtype: object
```
Eliminamos los registros con formaato dde fecha  "SOON"
```python
archivo1_dataframe= archivo1_dataframe.drop([101344,120318])
```
Del archivo australian_user_reviews.json tenemos: 
```python
import ast
lista = [ ]
with open("../models/australian_user_reviews.json",encoding='utf-8') as f:
    for line in f.readlines():
        lista.append(ast.literal_eval(line))
```

```python
df = pd.DataFrame(lista)
df.head(5)
```
para el análisis de sentimientos importamos la libreria TextBlob y hacemos el análisis
```python
from textblob import TextBlob 

def sentiment_analysis(review):
    if isinstance(review, list) and review:
        text = review[0].get('review','')
        sentiment = TextBlob(text).sentiment.polarity
        if sentiment < -0.2:
            return 0 # Negative
        elif -0.2 <= sentiment <= 0.2:
            return 1 # Neutral
        else:
            return 2 # Positive
    else:
        return 1
```
Se crea una lista vacia paraa guardas los datos desanidados 
```python
import pandas as pd

# Definir una función que se aplicará a cada fila del DataFrame
def desanidar_fila(row):
    user_id = row['user_id']
    user_url = row['user_url']
    sentiment_analysis_value = row['sentiment_analysis']
    reviews = row['reviews']

    # Verificar si reviews es una lista antes de intentar desanidar
    if isinstance(reviews, list):
    # Crear una lista de diccionarios desanidados para cada elemento en la lista de reviews
        desanidados = [{
            'user_id': user_id,
            'user_url': user_url,
            'reviews': reviews,
            'sentiment_analysis': sentiment_analysis_value,
            'posted': e.get('posted', ''),
            'item_id': e.get('item_id', ''),
            'recommend': e.get('recommend', False),
            'review': e.get('review', '')
        } for e in reviews]
    elif isinstance(reviews, float) and not pd.notna(reviews):
        # Si reviews es NaN, asignar una lista vacía a desanidados
        desanidados = []

    return desanidados

```
Ahora aplicamos  la función a cada fila del DataFrame y convertir la lista de listas en una lista plana
```python
lista_desanidada = df.apply(desanidar_fila, axis=1).explode().tolist()
```
para probar 
```python
df_desanidado = pd.DataFrame(lista_desanidada)
df.drop('reviews', axis=1, inplace=True)

print(df_desanidado)
```
  -
**Desarrollo de la API**

**Endpoint 1:  /developer**
Este endpoint devuelve la cantidad de items y porcentaje de contenido Free por año según empresa desarrolladora.
- **Metodo**: GET
- **'URL'**:https://machine-learning-operations-byli.onrender.com/api/v1/developer/{developer}
- **Parámetros de consulta:** developer

**Endpoint 2:  /user**
Este endpoinnt  devuelve la cantidad de dinero gastado por el usuario, el porcentaje de recomendación en base a reviews.recommend y cantidad de items
- **Metodo**: GET
- **URL:**https://machine-learning-operations-byli.onrender.com/api/v1/userdata/{developer}
- **Parámetros de consulta:** user

**Endpoint 3:  /genre**
 Este endpoint  devuelve el usuario que acumula más horas jugadas para el género dado y una lista de la acumulación de horas jugadas por año de lanzamiento.
- **Metodo**: GET
- **URL:**https://machine-learning-operations-byli.onrender.com/api/v1/user-for-genre/{genre}
- **Parámetros de consulta:** genre

**Endpoint 4:  /year**
Este endpoint devuelve el top 3 de desarrolladores con juegos MÁS recomendados por usuarios para el año dado. (reviews.recommend = True y comentarios positivos
- **Metodo**: GET
- **URL:**https://machine-learning-operations-byli.onrender.com/api/v1/best-developer-year/{year}
- **Parámetros de consulta:** year

**Endpoint 5:  /developer**
Este endpoint  devuelve un diccionario con el nombre del desarrollador como llave y una lista con la cantidad total de registros de reseñas de usuarios que se encuentren categorizados con un análisis de sentimiento como valor positivo o negativo.
- **Metodo**: GET
- **URL:**https://machine-learning-operations-byli.onrender.com/api/v1/developer-reviews-analysis/{developer}
- **Parámetros de consulta:** developer