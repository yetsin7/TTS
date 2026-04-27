# XTTS Fine-Tuning

Este proyecto ya no debe apoyarse en el modelo experimental inicial para calidad real.  
La ruta recomendada es **fine-tuning de XTTS-v2** sobre tu propia voz.

## Decision técnica

- Recomendado para tu caso: **Google Colab con GPU**
- No recomendado para este fine-tuning: tu laptop actual en CPU
- La GPU de tu equipo aparece en `nvidia-smi`, pero `PyTorch` local esta en modo CPU y 6 GB de VRAM no es una base comoda para XTTS-v2

## Fuentes oficiales

- Documentación oficial de XTTS: [Coqui XTTS docs](https://tts.readthedocs.io/en/dev/models/xtts.html)
- Repositorio oficial: [coqui-ai/TTS](https://github.com/coqui-ai/TTS)
- Receta oficial usada como base: `recipes/ljspeech/xtts_v2/train_gpt_xtts.py`
- Modelo oficial: [coqui/XTTS-v2](https://huggingface.co/coqui/XTTS-v2)

## Flujo recomendado

1. Coloca tu audio y tu transcript en:
   - `C:\Dev\TTS\data\raw\audio_entrenamiento.wav`
   - `C:\Dev\TTS\data\raw\texto_entrenamiento.txt`
2. Prepara el dataset base:
   - `python scripts/run_pipeline.py`
3. Exporta dataset para XTTS:
   - `python scripts/export_xtts_dataset.py`
4. Sube `C:\Dev\TTS\data\xtts` a Google Drive o al entorno de Colab.
5. En Colab instala dependencias:
   - `pip install -r requirements-xtts-colab.txt`
6. Corre:
   - `python scripts/train_xtts.py`
7. Para inferencia:
   - `python scripts/infer_xtts.py --text "Hola, esta es una prueba con XTTS."`

## Dataset esperado por XTTS

El exportador genera:

- `data/xtts/wavs/*.wav`
- `data/xtts/metadata.csv`

Formato de `metadata.csv`:

```text
clip_0000|texto normalizado|texto normalizado
clip_0001|texto normalizado|texto normalizado
```

## Nota importante

Tu dataset actual fue creado con `even_split_fallback`, o sea con segmentacion uniforme de audio.
Eso sirve para prototipar, pero puede limitar bastante la calidad.
Antes de un fine-tuning serio, conviene mejorar la alineacion real frase-a-frase.
