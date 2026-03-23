# Archivos de Entrenamiento

Guarda aqui tus archivos crudos antes de ejecutar el entrenamiento.

## Rutas exactas aceptadas

- `C:\Dev\TTS\data\raw\audio_entrenamiento.wav`
- `C:\Dev\TTS\data\raw\audio_entrenamiento.mp3`
- `C:\Dev\TTS\data\raw\audio_entrenamiento.flac`
- `C:\Dev\TTS\data\raw\texto_entrenamiento.txt`

## Formato recomendado

- Recomendado: `audio_entrenamiento.wav`
- Voz de una sola persona
- Grabacion limpia, sin musica ni ruido
- Una sola sesion o varias sesiones con el mismo microfono
- Pausa breve entre frases
- Minimo util: 30 a 60 minutos
- Mejor resultado: 2 a 10 horas

## Formato del texto

Escribe una frase por linea en `texto_entrenamiento.txt`.

Ejemplo:

```text
Hola, esta es una prueba de mi voz.
Estoy grabando frases cortas y limpias para entrenar el modelo.
Este sistema usara mi propia voz para leer textos nuevos.
```

## Flujo

1. Guarda el audio con uno de los nombres permitidos.
2. Guarda el texto como `texto_entrenamiento.txt`.
3. Ejecuta `python scripts/preflight.py`.
4. Ejecuta `python scripts/run_pipeline.py`.
