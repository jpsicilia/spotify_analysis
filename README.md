# Analisisi de datos: Spotify 
## Descripcion del Proyecto: 
A través de este análisis, buscamos responder preguntas clave sobre la industria musical moderna, con especial foco en tendencias de géneros, impacto del contenido explícito, evolución temporal de características musicales y patrones de engagement de artistas.

## Dataset
- **Fuente:** Kaggle  
- **Descripción:**
| Variable         | Descripción                                 |
|------------------|---------------------------------------------|
| artist           | Artista                                     |
| song             | Nombre de la canción                       |
| duration_ms      | Duración en milisegundos                   |
| explicit         | Si es explícita (True/False)               |
| year             | Año de lanzamiento                         |
| popularity       | Popularidad (0 a 100)                      |
| danceability     | Qué tan bailable es (0-1)                  |
| energy           | Nivel de energía (0-1)                     |
| loudness         | Volumen medio (dB)                         |
| speechiness      | Presencia de palabras habladas (0-1)       |
| acousticness     | Nivel de acústica (0-1)                    |
| instrumentalness | Probabilidad de ser instrumental (0-1)     |
| liveness         | Probabilidad de grabación en vivo (0-1)    |
| valence          | Positividad o felicidad (0-1)              |
| tempo            | Tempo (BPM)                                |
| genre            | Géneros asociados (puede ser múltiple)     |

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
