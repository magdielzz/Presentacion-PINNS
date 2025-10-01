# Presentación: PINNs para la Ecuación de Schrödinger No Lineal

Sitio web interactivo para compartir la presentación sobre **Physics-Informed Neural Networks (PINNs)** aplicadas a la resolución de la Ecuación de Schrödinger No Lineal (NLSE).

## 📋 Descripción

Esta presentación explora el uso de **Redes Neuronales Informadas por la Física (PINNs)** como una alternativa innovadora a los métodos numéricos tradicionales para resolver ecuaciones diferenciales parciales. El proyecto se enfoca específicamente en la **Ecuación de Schrödinger No Lineal**, demostrando cómo las PINNs pueden resolver problemas complejos de física sin necesidad de mallas computacionales.

### ¿Qué son las PINNs?

Las PINNs son redes neuronales que incorporan ecuaciones físicas (PDEs) directamente en la función de pérdida durante el entrenamiento. Esto permite que el modelo aprenda soluciones que respetan las leyes fundamentales de la física.

## ✨ Características Principales

- **Sin malla**: Maneja geometrías complejas sin necesidad de discretización espacial
- **Incorpora leyes de conservación**: Las ecuaciones físicas se integran en el proceso de aprendizaje
- **Robusto ante ruido**: Capacidad de trabajar con datos ruidosos
- **Alta flexibilidad**: Adaptable a diferentes tipos de problemas físicos
- **Precisión comparable**: Resultados de alta calidad comparables con métodos tradicionales

## 🛠️ Tecnologías Utilizadas

### Frameworks Comparados

La presentación incluye una comparativa exhaustiva de tres frameworks populares:

| Framework | Error Final | Tiempo (s) | Velocidad Relativa |
|-----------|-------------|------------|-------------------|
| TensorFlow 1 | 0.00278 | 5023.85 | Más lento |
| TensorFlow 2 | **0.00145** | 1546.84 | Medio |
| PyTorch | 0.00301 | **369.10** | **Más rápido** |

**Conclusión**: PyTorch fue el más rápido en entrenamiento, mientras que TensorFlow 2 obtuvo el menor error final.

### Herramientas de Visualización

- **Reveal.js**: Framework para presentaciones interactivas
- **Chart.js**: Visualización de gráficos y métricas
- **Plotly**: Gráficos 3D interactivos
- **Tailwind CSS**: Diseño responsive y moderno
- **MathJax**: Renderizado de ecuaciones matemáticas

## 📊 Estructura de la Presentación

La presentación está organizada en las siguientes secciones:

1. **Introducción**: Motivación y contexto del problema
2. **Fundamentos de PINNs**: Conceptos básicos y arquitectura
3. **Ecuación de Schrödinger No Lineal**: Formulación del problema
4. **Implementación**: Detalles técnicos de la red neuronal
5. **Flujo de Entrenamiento**: 
   - Preparación (generación de puntos de colocación y contorno)
   - Optimización Adam (10,000 iteraciones)
   - Optimización L-BFGS-B (refinamiento final)
   - Evaluación y validación
6. **Resultados**: Métricas de rendimiento y visualizaciones
7. **Comparativa de Frameworks**: TensorFlow 1, TensorFlow 2 y PyTorch
8. **Visualización de Soluciones**: Gráficos interactivos para N=1 y N=2
9. **Conclusiones y Referencias**

## 🔬 Resultados Destacados

### Métricas de Rendimiento

- **Error Relativo L₂** en |h(x,t)|:
  - TensorFlow 1: 2.78×10⁻³
  - TensorFlow 2: 1.45×10⁻³ ⭐
  - PyTorch: 3.01×10⁻³

### Ventajas Demostradas

- ✅ Precisión alta en la predicción de soluciones
- ✅ Capacidad de manejar geometrías complejas
- ✅ Integración natural de condiciones de contorno
- ✅ Potencial para acelerar simulaciones científicas

### Limitaciones Identificadas

- ⚠️ Tiempo de entrenamiento elevado
- ⚠️ Sensibilidad a la elección de hiperparámetros
- ⚠️ Requiere conocimiento de la física subyacente

## 🚀 Uso

### Visualizar la Presentación

1. Clona este repositorio:
```bash
git clone https://github.com/magdielzz/Presentacion-PINNS.git
cd Presentacion-PINNS
```

2. Abre los archivos HTML en tu navegador:
```bash
# Archivo principal de presentación
open indec.html
# o
open presentacion2.html
```

3. Navega por las diapositivas usando:
   - **Flechas ← →**: Diapositiva anterior/siguiente
   - **Flechas ↑ ↓**: Navegación vertical (sub-diapositivas)
   - **Esc**: Vista general de diapositivas
   - **F**: Pantalla completa

### Generar Datos de Entrenamiento

La presentación incluye código Python para generar archivos `.mat` con soluciones de la ecuación de Schrödinger:

```python
import numpy as np
import scipy.io as sio

def nonlinear_schrodinger_solution(N=1):
    """
    Genera solución exacta de la ecuación de Schrödinger no lineal
    para N solitones
    """
    # Ver código completo en la presentación
    pass

# Generar solución para N=1
data_n1 = nonlinear_schrodinger_solution(N=1)

# Generar solución para N=2
data_n2 = nonlinear_schrodinger_solution(N=2)
```

## 📈 Comparación con Métodos Tradicionales

| Método | Precisión | Flexibilidad | Requiere Malla |
|--------|-----------|--------------|----------------|
| Diferencias Finitas | Media | Baja | ✓ |
| Elementos Finitos | Alta | Media | ✓ |
| **PINNs** | **Alta** | **Muy Alta** | ✗ |

## 🎓 Referencias y Recursos

### Artículos Clave

- Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). *Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations*. Journal of Computational Physics.

### Enlaces Útiles

- [Repositorio Original de PINNs - Maziar Raissi](https://github.com/maziarraissi/PINNs)
- [Physics-Informed Neural Networks (PINNs): An Intuitive Guide](https://towardsdatascience.com/physics-informed-neural-networks-pinns-an-intuitive-guide-fff138069563)

## 📝 Contenido Técnico

### Arquitectura de la Red

- **Capas**: Red neuronal profunda totalmente conectada
- **Inicialización**: Xavier (para evitar gradientes que se desvanecen/explotan)
- **Entrada**: Tensor `[x, t]` normalizado a `[-1, 1]`
- **Salida**: Tensor `[u(x,t), v(x,t)]` (partes real e imaginaria)
- **Función de Activación**: tanh (típicamente)

### Función de Pérdida

La función de pérdida combina tres componentes:

1. **Pérdida PDE**: Residuo de la ecuación diferencial en puntos de colocación
2. **Pérdida de Condiciones Iniciales**: Error en t=0
3. **Pérdida de Condiciones de Contorno**: Error en los bordes del dominio

## 🤝 Contribuciones

Este proyecto es una presentación educativa sobre PINNs. Si encuentras errores o tienes sugerencias para mejorar el contenido, siéntete libre de abrir un issue o pull request.

## 📧 Contacto

Para preguntas o comentarios sobre esta presentación, por favor abre un issue en este repositorio.

## 📄 Licencia

Este proyecto está disponible como recurso educativo. Por favor, cita adecuadamente si utilizas este material en tu trabajo.

---

**Nota**: Esta presentación fue creada con fines educativos para demostrar las capacidades de las Physics-Informed Neural Networks en la resolución de ecuaciones diferenciales parciales.
