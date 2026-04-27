# TTS

Proyecto en Python para construir un sistema completo de sintesis de voz desde cero con una sola voz.

## Camino recomendado en 2026

Para una voz realmente útil, este repo ya incluye una ruta mas realista:

- pipeline base local para preparar datos
- exportacion a dataset para `XTTS-v2`
- fine-tuning recomendado en Google Colab

Consulta [XTTS_FINE_TUNING.md](XTTS_FINE_TUNING.md).

## Objetivo

Esta base te permite:

- Guardar tu audio y tu texto en una estructura fija.
- Preparar y validar un dataset simple para una sola voz.
- Entrenar un modelo inicial de TTS desde cero.
- Generar audio nuevo con tu voz.
- Probar el modelo en una interfaz local.
- Exponer el modelo por API HTTP local.
- Convertir `mp3`, `wav` o `flac` a un WAV normalizado para entrenamiento.

## Importante

Aunque el proyecto acepta un unico archivo `audio_entrenamiento.wav`, `audio_entrenamiento.mp3` o `audio_entrenamiento.flac` junto con `texto_entrenamiento.txt`, para obtener una voz util necesitas muchas frases.

- Recomendado mínimo: 30 a 60 minutos de audio limpio.
- Mejor resultado: 2 a 10 horas de audio limpio.
- El texto debe estar dividido una frase por linea y el audio debe incluir pausas claras entre frases.
- El sistema normaliza el audio y guarda una copia en `data/processed/audio_normalizado.wav`.

## Estructura

```text
data/
  raw/
    audio_entrenamiento.wav
    texto_entrenamiento.txt
  processed/
models/
outputs/
src/tts_project/
scripts/
```

## Instalación

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -U pip
pip install -e .
```

## Flujo

1. Coloca tu audio en `C:\Dev\TTS\data\raw\audio_entrenamiento.wav`, `C:\Dev\TTS\data\raw\audio_entrenamiento.mp3` o `C:\Dev\TTS\data\raw\audio_entrenamiento.flac`.
2. Coloca tu texto en `C:\Dev\TTS\data\raw\texto_entrenamiento.txt`.
3. Ejecuta `python scripts/preflight.py`.
4. Ejecuta `python scripts/run_pipeline.py`.
5. Ejecuta `python scripts/synthesize.py --text "Hola, esta es mi voz."`.
6. Si quieres interfaz, ejecuta `python app.py`.
7. Si quieres API, ejecuta `python scripts/serve_api.py`.

## Componentes del sistema

- `scripts/prepare_dataset.py`: convierte, normaliza, segmenta el audio crudo y genera el manifiesto.
- `scripts/preflight.py`: confirma que las rutas y el formato esten listos.
- `scripts/inspect_dataset.py`: valida que el corpus sea consistente.
- `scripts/train.py`: entrena el modelo principal.
- `scripts/export_xtts_dataset.py`: convierte el dataset actual al formato de XTTS.
- `scripts/validate_xtts_dataset.py`: valida el dataset exportado contra límites útiles para XTTS.
- `scripts/train_xtts.py`: lanza fine-tuning de XTTS-v2 en un entorno con GPU y Coqui TTS.
- `scripts/infer_xtts.py`: prueba inferencia con XTTS.
- `scripts/synthesize_best.py`: usa XTTS fine-tuned si existe; si no, cae al modelo experimental.
- `scripts/package_xtts_bundle.py`: crea un zip mínimo para llevar el flujo a Colab.
- `scripts/synthesize.py`: sintetiza audio desde consola.
- `scripts/serve_api.py`: levanta una API local con FastAPI.
- `app.py`: interfaz local de escritorio para Windows.
- `src/tts_project/service.py`: servicio central reutilizable de inferencia.
- `src/tts_project/trainer.py`: motor modular de entrenamiento.

## Notas técnicas

- Se usa un pipeline moderno sobre `PyTorch` y `torchaudio`.
- El modelo inicial es deliberadamente compacto para poder iterar rápido.
- La reconstrucción de audio usa Griffin-Lim como base inicial.
- Después podremos sustituir el vocoder por uno neuronal mas avanzado.
- La arquitectura actual esta pensada para que luego podamos migrar a un stack mas potente sin rehacer el proyecto.
- Si el segmentado por silencios no coincide con el texto, el sistema usa un respaldo por particion uniforme para no bloquear el pipeline.
- El ruido actual proviene del modelo experimental base. La ruta recomendada para mejor calidad es XTTS fine-tuned.

## Resumen operativo

- Ruta de audio recomendada: `C:\Dev\TTS\data\raw\audio_entrenamiento.wav`
- Ruta de texto obligatoria: `C:\Dev\TTS\data\raw\texto_entrenamiento.txt`
- Comando unico para entrenar desde datos crudos: `python scripts/run_pipeline.py`
