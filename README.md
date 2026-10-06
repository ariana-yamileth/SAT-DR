# SAT-DR

## Descripción
Proyecto de aprendizaje automático. Genera datos sintéticos (`src/generar_datos.py`) y entrena un modelo MLP con scikit-learn (`src/entrenar.py`), ajustando el umbral de decisión para lograr un recall >= 85 %. [Agrega una frase sobre qué problema resuelve y qué predice el modelo.]

## Ejecución
Requiere Python 3.14.6 (o compatible) y Git.

    git clone https://github.com/ariana-yamileth/sat-dr.git
    cd sat-dr
    py -m venv .venv
    .\.venv\Scripts\Activate.ps1
    pip install -r requirements.txt
    py src/generar_datos.py
    py src/entrenar.py

Los datos quedan en `data/` y el modelo en `models/`. [Confirma las rutas de salida.]

## Versiones
| Librería | Versión |
|----------|---------|
| Python | 3.14.6 |
| streamlit | 1.65.0 |
| scikit-learn | 1.9.1 |
| pandas | 3.0.6 |
| numpy | 2.5.3 |
| joblib | 1.6.0 |

Las versiones definitivas quedan fijadas en `requirements.txt`.

## Créditos
| Integrante | Aportes |
|-----------|---------|
| [Nombre 1] | [Qué hizo] |
| [Nombre 2] | [Qué hizo] |
| [Nombre 3] | [Qué hizo] |

## Código de IA vs. aportes del equipo
- **Generado o asistido por IA:** [Partes y herramienta usada.]
- **Aportes del equipo:** [Lo escrito, revisado, probado o decidido por el equipo.]

El registro de prompts está en [prompts/bitacora.md](prompts/bitacora.md).

## Licencia
[Licencia elegida, p. ej. MIT.] Las librerías de terceros se usan bajo sus propias licencias de código abierto.
