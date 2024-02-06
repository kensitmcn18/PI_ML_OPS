import os
import json
from fastapi import APIRouter
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from textblob import TextBlob

# Obtén la ruta al directorio 'models'
models_directory = os.path.join(os.path.dirname(__file__), '../models')
ruta_archivo_user_items_parquet = os.path.join(models_directory, 'user_items.parquet')
ruta_archivo_user_reviews_parquet = os.path.join(models_directory, 'user_reviews.parquet')
ruta_archivo_steam_games_parquet = os.path.join(models_directory, 'steam_games.parquet')

router = APIRouter()

@router.get("/developer/{developer}")
def developer(developer : str):

    try:
        df = pd.read_parquet(ruta_archivo_steam_games_parquet, columns=['release_date', 'price', 'developer'], engine='pyarrow')

        df_publisher = df[df['developer'] == developer].copy(deep=True)

        df_publisher['release_date'] = pd.to_datetime(df_publisher['release_date'], errors='coerce')

        # Crear una columna adicional para indicar si el contenido es gratis
        df_publisher['es_gratis'] = df_publisher['price'].isin([0.0])

        # Agrupar por año y contar la cantidad de items y el porcentaje de contenido gratis
        resultado = df_publisher.groupby(df_publisher['release_date'].dt.year)['es_gratis'].agg(['count', 'mean']).reset_index()

        resultado['mean'] = (resultado['mean'] * 100).round(2)

        resultado = resultado.rename(columns={'release_date': 'year', 'count': 'number_of_items', 'mean': 'free_content'})
        
        return convertir_dataframe_a_json(resultado)

    except Exception as e:
        print(f"Ocurrió una excepción developer: {e}")
        return None
    

@router.get("/userdata/{user_id}")
def userdata(user_id : str):                                     
    
    try:
        user_items_df  = pd.read_parquet(ruta_archivo_user_items_parquet, columns=['user_id', 'item_id', 'item_name'], engine='pyarrow')
        user_reviews_df  = pd.read_parquet(ruta_archivo_user_reviews_parquet, columns=['user_id', 'item_id', 'recommend'], engine='pyarrow')
        steam_games_df = pd.read_parquet(ruta_archivo_steam_games_parquet, columns=['id', 'price'], engine='pyarrow')

        # Realizar un merge para obtener los precios correspondientes
        merged_df = pd.merge(user_items_df, steam_games_df, left_on='item_id', right_on='id', how='inner')
        
        # Filtrar por user_id
        user_items_filtered = user_items_df[user_items_df['user_id'] == user_id]
        user_reviews_filtered = user_reviews_df[user_reviews_df['user_id'] == user_id]
        merged_df_filtered = merged_df[merged_df['user_id'] == user_id]

        # Calcular la cantidad de dinero gastado por el usuario
        money_spent = merged_df_filtered['price'].sum() if not merged_df_filtered.empty else 0

        # Calcular el porcentaje de recomendación en base a reviews.recommend
        recommendation_percentage = user_reviews_filtered['recommend'].mean() * 100 if not user_reviews_filtered.empty else 0

        # Calcular la cantidad de items
        items_count = len(user_items_filtered)

        result =  {
            "user": user_id,
            "total_money_spent": round(float(money_spent),2),
            "recommendation_percentage": recommendation_percentage,
            "total_items": int(items_count)
        }
        
        return result
    except Exception as e:
        print(f"Ocurrió una excepción userdata: {e}")
        return None

@router.get("/user-for-genre/{genre}")
def userForGenre(genre : str):

    try:
        user_items_df  = pd.read_parquet(ruta_archivo_user_items_parquet, engine='pyarrow')
        steam_games_df = pd.read_parquet(ruta_archivo_steam_games_parquet, engine='pyarrow')

        # Filtrar juegos por género
        genre_games = steam_games_df[steam_games_df['genres'].apply(lambda x: genre in x)]

        # Combinar dataframes
        merged_df = pd.merge(user_items_df, genre_games, left_on='item_id', right_on='id', how='inner')

        # Encontrar usuario con más horas jugadas para el género dado
        user_with_more_hours = merged_df.groupby('user_id')['playtime_forever'].sum().idxmax()

        # Crear lista de acumulación de horas jugadas por año de lanzamiento
        merged_df['release_date_year'] = pd.to_datetime(merged_df['release_date']).dt.year
        hours_played_by_year = merged_df.groupby('release_date_year')['playtime_forever'].sum().reset_index()
        hours_played_by_year = hours_played_by_year.rename(columns={'release_date_year': 'year', 'playtime_forever': 'hours'})
        hours_played_by_year = hours_played_by_year.sort_values('year', ascending=False)
        hours_played_list = hours_played_by_year.to_dict(orient='records')

        # Crear el diccionario con los resultados
        result = {
            "user_with_more_hours_played_for_{}".format(genre): user_with_more_hours,
            "hours_played": hours_played_list
        }

        return result
    except Exception as e:
            print(f"Ocurrió una excepción userForGenre: {e}")
            return None

@router.get("/best-developer-year/{year}") 
def bestDeveloperYear (year : int):
    try:
        # user_items_df  = pd.read_parquet(ruta_archivo_user_items_parquet, engine='pyarrow')
        steam_games_df = pd.read_parquet(ruta_archivo_steam_games_parquet, engine='pyarrow')
        user_reviews_df  = pd.read_parquet(ruta_archivo_user_reviews_parquet, engine='pyarrow')

        # Combinar dataframes
        # merged_df = pd.merge(user_items_df, user_reviews_df, on='item_id', how='inner')
        merged_df = pd.merge(user_reviews_df, steam_games_df, left_on='item_id', right_on='id', how='inner')

        # Filtrar por el año proporcionado
        juegos_por_anio = merged_df[merged_df['release_date'].str.contains(str(year))]

        # Calcular el top 3 de desarrolladores con juegos más recomendados
        top_desarrolladores = juegos_por_anio.groupby('developer')['recommend'].mean().sort_values(ascending=False).head(3)

        # Crear la lista de resultados
        resultados = [{"Rank {}".format(i+1): desarrollador} for i, (desarrollador, _) in enumerate(top_desarrolladores.items())]

        return resultados
    except Exception as e:
            print(f"Ocurrió una excepción userdata: {e}")
            return None
@router.get("/developer-reviews-analysis/{developer}")
def developerReviewsAnalysis(developer : str):
    # Cargar los datos desde los archivos .parquet
    steam_games_df = pd.read_parquet(ruta_archivo_steam_games_parquet, engine='pyarrow')
    user_reviews_df  = pd.read_parquet(ruta_archivo_user_reviews_parquet, engine='pyarrow')

    # Combinar dataframes
    merged_df = pd.merge(user_reviews_df, steam_games_df, left_on='item_id', right_on='id', how='inner')

    # Filtrar por el desarrollador proporcionado
    desarrollador_df = merged_df[merged_df['developer'] == developer].copy()

    # Realizar el análisis de sentimiento y contar las reseñas positivas y negativas
    desarrollador_df['sentimiento'] = desarrollador_df['review'].apply(analisis_sentimiento)
    conteo_sentimientos = desarrollador_df['sentimiento'].value_counts().to_dict()

    # Crear el diccionario con los resultados
    resultados = {developer: {'Positive': conteo_sentimientos.get('Positive', 0),
                                  'Negative': conteo_sentimientos.get('Negative', 0)}}

    return resultados

def analisis_sentimiento(texto):
    analysis = TextBlob(texto)
    return 'Positive' if analysis.sentiment.polarity > 0 else 'Negative'

def convertir_dataframe_a_json(df):
    dataframe_dict = df.to_dict(orient='records')
    dataframe_json = json.dumps(dataframe_dict, indent=2)
    dataframe_json_sin_formato = ''.join(dataframe_json.split())
    dataframe_objeto = json.loads(dataframe_json_sin_formato)
    return dataframe_objeto