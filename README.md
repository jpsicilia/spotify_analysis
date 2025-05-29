# Analisisi de datos: Spotify 
## Descripcion del Proyecto: 
A través de este análisis, buscamos responder preguntas clave sobre la industria musical moderna, con especial foco en tendencias de géneros, impacto del contenido explícito, evolución temporal de características musicales y patrones de engagement de artistas.

## Dataset
- **Fuente:** Kaggle  
- **Descripción:**
  
| Variable         | Descripción                                          |
|------------------|------------------------------------------------------|
| artist           | Artista                                              |
| song             | Nombre de la canción                                 |
| duration_ms      | Duración de la canción en milisegundos               |
| explicit         | Indica si la canción tiene contenido explícito       |
| year             | Año de lanzamiento                                   |
| popularity       | Popularidad de la canción en Spotify (0 a 100)       |
| danceability     | Qué tan bailable es la canción (0 a 1)               |
| energy           | Nivel de energía de la canción (0 a 1)               |
| key              | Tono musical representado como un entero             |
| loudness         | Volumen medio de la canción en decibelios (dB)       |
| mode             | Modalidad (1=Mayor, 0=Menor)                         |
| speechiness      | Presencia de palabras habladas en la canción (0 a 1) |
| acousticness     | Confianza de que la canción sea acústica (0 a 1)     |
| instrumentalness | Probabilidad de que sea instrumental (0 a 1)         |
| liveness         | Probabilidad de que sea una grabación en vivo (0 a 1)|
| valence          | Medida de positividad o felicidad percibida (0 a 1)  |
| tempo            | Tempo de la canción en beats por minuto (BPM)        |
| genre            | Género o géneros asociados                           |


## Tecnologías y Librerías
- Python 
- Pandas
- Numpy
- Matplotlib
- Seaborn
- Scikit-learn

## Highlights del Análisis

Limpieza y normalización: Detección y eliminación de duplicados, manejo de géneros inconsistentes.

Artistas con mayor engagement: Popularidad media ponderada por volumen de canciones.

Géneros premium: Exploración de qué combinaciones de géneros son dominantes en popularidad y cantidad.

Impacto de contenido explícito: Impacto del contenido explícito en la popilaridad.

Duración óptima: Análisis sobre qué tan largas son las canciones exitosas.

Red de géneros: Visualización de la co-ocurrencia de géneros con grafos.

Evolución temporal: Análisis de cómo han cambiado las características musicales a lo largo del tiempo.

## Cómo ejecutar el proyecto
1. Clona este repositorio:
```bash
git clone https://github.com/jpsicilia/spotify_analysis.git
cd spotify_analysis
```
3. Instala las dependencias:
pip install -r requirements.txt
