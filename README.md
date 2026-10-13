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
|Ariana Yamileth |Desarrollé el proyecto SAT-DR, desde la planeación y distribución de actividades hasta la creación del repositorio, revisión de Pull Requests, resolución de conflictos y organización del historial de Git. Mejoré la interfaz, integré gráficas de riesgo, soporte para archivos Excel y tema oscuro. Finalmente, desplegué la aplicación en Streamlit Community Cloud y verifiqué su funcionamiento.
Uso de herramientas de IA: Utilicé Gemini para resolver dudas sobre Git y GitHub, y Claude para apoyar la planeación, arquitectura y desarrollo de la aplicación, incluyendo las gráficas, la configuración del tema y la compatibilidad con Excel. Ejecuté y verifiqué los comandos y cambios realizados en el proyecto.|
|Dulce valeria| Usé Claude para entender la rama de redes neuronales, elegir el caso de uso (riesgo de abandono escolar) y el stack, y para que me explicara cómo armar el generador de datos y el entrenamiento. La IA generó el código base de generar_datos.py y entrenar.py. Yo descarté el reconocimiento facial por privacidad y dificultad, corregí la parte del código que no coincidía y ajusté el tamaño de la muestra de 2,000 a 1,500 estudiantes. Reentrené el modelo en mi equipo y obtuve umbral 0.23, Recall 87.8 %, precisión 58.9 % y exactitud 76 %. Hice los commits, subí la rama y abrí el PR con Ariana como reviewer.|
|Rosario Monserrat|Usé Claude para desarrollar la interfaz en Streamlit que predice el riesgo de deserción escolar. La IA generó el código base de las pestañas "Estudiante individual" (sliders, probabilidad, semáforo y recomendación) y "Rendimiento del modelo" (recall, precisión, exactitud y matriz de confusión), y me orientó con Git. Yo probé la interfaz con distintos valores, verifiqué que las métricas coincidieran con metricas.json y ajusté el diseño y los textos. Hice los commits, subí la rama feat/app-individual y abrí el PR "feat: interfaz individual y métricas" con reviewer asignado. Además, elaboré el Manual de usuario de SAT-DR v1.1, dirigido a personas sin conocimientos técnicos, con la ejecución paso a paso, la interpretación de resultados, preguntas frecuentes y solución de problemas. También hice la prueba con una persona ajena (una maestra de primaria, en la que observé como QA sus dificultades: no identificó el botón "Upload", no entendió la "probabilidad de riesgo" y tardó en encontrar el archivo descargado. Con esos hallazgos corregí el manual: agregué una captura que marca el botón, aclaré que 0.22 no es una calificación y puse un ejemplo con porcentajes, e indiqué que el archivo queda en Descargas y que Ctrl + J abre la lista de descargas|
|Hugo Garcia|Yo realice el archivo de bitacora de promts y readme para llenarlo con la infomacion recabada del equipo de desarrollo, tambien realice las pruebas de nuestro proyecto como validar que los datos de entrada fueran los que este solicita y que los resultados fueran los que esperamos y validar el funcionamiento al igual que hallar detalles para notificar las posibles mejoras o agregarlas al archivo de Casos qa ya que tambien |

## Código de IA vs. aportes del equipo
- **Generado o asistido por IA:** [Partes y herramienta usada.]
- **Aportes del equipo:** [Lo escrito, revisado, probado o decidido por el equipo.]

El registro de prompts está en [prompts/bitacora.md](prompts/bitacora.md).

## Licencia
[Licencia elegida, p. ej. MIT.] Las librerías de terceros se usan bajo sus propias licencias de código abierto.
