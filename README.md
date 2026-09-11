# Mini-GPT en PyTorch — Baseline serio

> Un Transformer decoder entrenado **desde cero** en PyTorch puro, sin dependencias
> externas más allá de `torch`. Pensado para correr de principio a fin en
> **Google Colab Free (GPU T4)** en ~5-10 minutos.

Este proyecto es un *baseline* educativo: cubre el flujo completo de ML — dataset,
tokenización, modelo, entrenamiento, evaluación, generación y persistencia — en un
notebook único y comentado en español. No pretende competir con LLMs modernos; su
objetivo es ser **legible, ejecutable y extensible**.

---

## ¿Qué hay aquí?

- `mini_gpt_pytorch.ipynb` — Notebook único, listo para Colab, con 10 secciones.
- Arquitectura: Transformer decoder (~1.5M parámetros).
- Dataset: **Tiny Shakespeare** (~1 MB, se descarga automáticamente).
- Tokenización: char-level (simple y robusta para un baseline).
- Inferencia: soporta `temperature`, `top_k` y `top_p`.

## Arquitectura del modelo

| Componente | Configuración |
|------------|---------------|
| Block size (contexto) | 128 tokens |
| Capas (`n_layer`) | 4 |
| Cabezas de atención (`n_head`) | 4 |
| Dim. embedding (`n_embd`) | 128 |
| Dropout | 0.1 |
| Parámetros totales | ~1.5M |
| Vocabulario | 65 caracteres |

Implementación modular:

- `MultiHeadSelfAttention` — Atención escalada con máscara causal.
- `FeedForward` — MLP de dos capas con GELU.
- `Block` — Transformer block con pre-LayerNorm y conexiones residuales.
- `MiniGPT` — Modelo completo: embedding posicional + N bloques + head lineal.

## Cómo ejecutarlo en Colab

1. Abre [Google Colab](https://colab.research.google.com/) y sube el notebook
   `mini_gpt_pytorch.ipynb` (File → Upload notebook).
2. Activa GPU: **Runtime → Change runtime type → T4 GPU**.
3. Ejecuta todo: **Runtime → Run all** (⌘/Ctrl + F9).
4. El entrenamiento tarda entre 5 y 10 minutos con GPU T4.
5. En la sección 8 del notebook hay una **celda interactiva** con sliders para
   generar texto con tu propio prompt.

> En CPU Colab también funciona, pero tarda entre 30 y 45 minutos.

## Resultados esperados

Tras 5 epochs sobre Tiny Shakespeare, el modelo produce texto que:

- Respeta mayúsculas, puntuación y saltos de línea.
- Genera nombres y frases en estilo shakesperiano.
- Aún no produce oraciones largas perfectamente coherentes — es un baseline
  de ~1.5M parámetros, no un LLM.

**Loss de validación esperada:** ~1.5 - 1.7
**Loss de un modelo uniforme (referencia):** `−ln(1/65) ≈ 3.83`

Ejemplo de generación con `temperature=0.8, top_k=40, top_p=0.9`:

```
PROMPT: "ROMEO:"
ROMEO:
I do protest I never injur'd thee,
But thou hast the valor of thy sword
Against my ruining: yet I will not...
```

## Próximos pasos

El notebook incluye una guía completa (sección 10) para evolucionar este baseline
hacia un chatbot conversacional:

1. **Cambiar el dataset** — DailyDialog u OpenSubtitles vía `datasets` de HuggingFace.
2. **Tokenización BPE** — con `tiktoken` (GPT-2 tokenizer) en lugar de char-level.
3. **Aumentar el modelo** — `n_layer=6, n_head=6, n_embd=384` (~10M params).
4. **Formato de chat** — mantener contexto con tags `<user>` / `<assistant>`.
5. **Fine-tuning eficiente** — PEFT / LoRA sobre modelos preentrenados (Qwen, LLaMA).

## Estructura del repositorio

```
mini-gpt/
├── README.md                  # Este archivo
├── LICENSE                    # MIT
├── .gitignore                 # Python + Jupyter
└── mini_gpt_pytorch.ipynb     # Notebook único (10 secciones)
```

## Cómo contribuir

Este es un proyecto educativo: **las contribuciones son bienvenidas**. Ideas:

- Mejoras en el notebook (comentarios, visualizaciones).
- Migración a BPE + dataset de diálogo (ver sección 10).
- Tests / métricas adicionales (perplexity, BLEU).
- Traducciones del README.

Pasos para contribuir:

1. Haz un fork del repositorio.
2. Crea una rama: `git checkout -b feat/mi-mejora`.
3. Haz tus cambios y commitea: `git commit -m "feat: descripción"`.
4. Envía un pull request.

> Si es tu primera vez contribuyendo a un proyecto open source, abre un issue
> presentándote y te guío paso a paso. En serio.

## Licencia

[MIT](./LICENSE) — libre para usar, modificar y distribuir.

## Agradecimientos

Inspirado en [nanoGPT](https://github.com/karpathy/nanoGPT) de Andrej Karpathy,
simplificado y comentado en español para fines educativos.
