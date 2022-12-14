# **Uso de framework de aprendizaje máquina**

_Author: Myroslava Sánchez Andrade_
<br>_Fecha de creación: 09/09/2022_
<br>_Última modificación: 14/09/2022_

---

## **Visión general**

El propósito de este repositorio es realizar la programación de un algoritmo de predicción.
<br>Las librerías necesarias para correr el código son: pandas, numpy, matplotlib, dataprep, sklearn y tensorflow.

Video de Google Colab:

---

## **Extract**

De la plataforma Kaggle se descargó un **[datset](https://www.kaggle.com/datasets/harlfoxem/housesalesprediction?select=kc_house_data.csv)** que contiene las casas vendidas entre Mayo del 2014 y Mayo del 2015 en King County. Está compuesto por las siguientes variables:

- **_Variables independientes:_** id (unic identifier for the house), date (date in which a house was put up for sale), bedrooms, bathroomns, sqft_living (living area inside the house), sqft_lot (total area of the property), floors, waterfront (sea view[boolean]), view (score), condition (score), grade (overall score), sqft_above (attic), sqft_basement, yr_built (year in which the house was built), year_renovated (last year of renovation), zipcode, lat (latitude), long (longitude), sqft_living15, sqft_lot15. Siendo categóricas las variables: `['view', 'condition', 'grade']`

- **_Variable dependiente:_** price (price at which a house was sold).

Para el anális de la estructura del dataset se utilizó la función **[create_report()](https://docs.dataprep.ai/user_guide/eda/create_report.html)** de la librería dataprep.eda que crea un reporte estadísitco descriptivo.

Después del análisis de los datos y su estructura se pudo identificar: variables que no aportaban a la secuencia de datos, se necesitban realizar normalización algunas variables y se necesitaba la reestructuración de una variable.

---

## **Transform**

Habiendo hecho un análisis de los datos, se realizaron las siguientes transformaciones:

- Transformación de la variable `['date']`: se hizo la separación del mes y del año de la fecha, no se tomó en cuenta el día, pues los datos sólo cotienen datos de un año.

- Selección de columnas: eliminado las columnas `['lat', 'long', 'zipcode']` porque no contribuyen al enriquecimiento del conjunto de datos ya que son variables de ubicación; para la relación de precio dada una ubicación, podemos usar las variables `sqft_living15` y `sqft_lot15` que dan información de las casas alrededor de una casa específica. La variable `id` también se eliminó ya que podemos usar el índice del DataFrame.

- Como se pudo observar en el reporte, en muchas de las variables independientes se tiene un rango de valores muy alto, por lo que se implementó una normalización. Esto evitará un posible sesgo.

- Ya que sólo se cuenta con un dataset para trabajar todo, se dividió este en 2 partes (Train/Test), para así poder llevar a cabo un correcto modelado; cada una de estas 2 partes, a su vez fue dividida en las variables independientes y la dependiente. (En el modelado se aplicó la validación)

---

## **Load**

Se exportaron los datasets ya limpios y transformados.

---

## **Model**

Para la modelación del problema planteado, se utilizaron 2 modelos: regresión multinear y red neuronal. Una vez se obtengan los resultados de ambos modelos y después de anlizarlos, se escogerá el mejor resultado para su regularización y ajuste para mejorar el rendimiento del modelo.

Para ambos modelos se utilizó la optimización de Adam, este es un método de descenso del gradiente que se basa en la estimación adaptativa.
<br>De igual manera, se utilizó una función de activación `Relu`. Se decidió usar esta función ya que es más simple, debido a que no activa a todas las neuronas al mismo tiempo.
<br>Por otro lado, se implementó una división del test del 0.2 para llevar a cabo la validación.

#### **_- Linear regression:_**

Para la aplicación de esta regresión lineal (y = mx + b), se produce una salida (output) al usar una sola capa y una neurona. Se utilizó un learning rate de 0.1 y un total de 5 épocas. Esto dio como resultado un error final (en la unidad de dólares) de 533585.8125, un bias promedio de 4.225e11 y una varianza promedio de 2436.365.

![image](https://user-images.githubusercontent.com/67491368/190276230-d2cf937c-6f2d-4498-bb2a-a629bfe3f032.png)


#### **_- Simple neural network:_**

Para este red neuronal simple, sólo se hizo uso de dos capas intermedias con 64 neuronas cada una; se usó la función de activación Relu, learning rate de 0.1 y 5 épocas. Esto dio como resultado un error final (en la unidad de dólares) de 120848.7188, un bias promedio de 3.952e10 y una varianza promedio de 7.453e8

![image](https://user-images.githubusercontent.com/67491368/190276241-15e24b48-a293-4ee1-b68b-f996764704d4.png)


---

## **Análisis de rendimiento**

Dados los resultados anteriores, se puede concluir (dado el decremento en el sesgo) que una red neuronal simple predice mejor la relación entre la variable independiente y las variables dependientes; y esto tiene sentido, ya que esperamos que el rendimiento de una red neuronal sea mejor que un modelo de regresión simple. A pesar de ello, la varianza aumentó, debido a la naturaleza menos agrupada de la red. Sin embargo, aun con la varianza aumentada, este es el mejor modelo por mucho una vez que revisamos el cambio en el error, por lo que fue el seleccionado para aplicar ajuste de parámetros.

- _Red neuronal simple (**fit**):_ 2 capas intermedias con 64 neuronas cada una.
    - Error: 120848.7188 dólares (**bajo**)
    - Sesgo: 3.952e10 (**bajo**)
    - Varianza: 7.453e8 (**bajo**)
    - A pesar de que los valores de error, sesgo y varianza pueden parecer altos, considerando el rango de valores de precios del dataset se ajustan relativamente bien al modelo. La predicción es buena sin llegar a un overfit, por lo tanto es un buen fit. A pesar de ello, intentaremos mejorar el rendimiento del modelo en la siguiente sección.

---

## **Regularización del modelo**

Teniendo la red neuronal como modelo para la predicción, ahora podemos aplicar técnicas de regularización o ajuste de parámetros para mejorar el rendimiento de la red neuronal.
<br>Se realizaron 3 cambios en los hiperparámetros:

- _Red neuronal compleja (aumentando el número de capas) (**underfit**):_ 4 capas intermedias con 40 neuronas cada una.
    - Error: 144444.0625 dólares (**medio**)
    - Sesgo: 3.586e10 (**bajo**)
    - Varianza: 2.758e9 (**alto**)
    - A pesar de que el sesgo es menor, la varianza y el error son mucho mayores que la red neuronal simple, lo cual implica un underfit en comparación con la misma.
- _Red neuronal simple con capa de abandono (para reducir el sobreajuste):_ 2 capas intermedias con una capa de droput de 0.1
    - Error: 121687.3594 dólares (**bajo**)
    - Sesgo: 4.187e10 (**medio**)
    - Varianza: 7.790e8 (**medio**)
    - A pesar de que la varianza y el error son comparables, el sesgo es mayor que la red neuronal simple lo cual implica un ligero overfit en comparación con la misma.
- _Red neuronal compleja con capas de abandono:_ 4 capas intermedias con 40 neuronas cada una y una capa de dropout de 0.1 y dos capas de dropout de 0.02
    - Error: 119325.7344 dólares (**bajo**)
    - Sesgo: 4.765e10 (**medio**)
    - Varianza: 2.137e9 (**alto**)
    - A pesar de que el error es menor, el sesgo y la varianza son mucho mayores que la red neuronal simple lo cual implica que este es un peor modelo en general.

---

## **Validación de resultados**

Comparando todos los modelos, podemos concluir que el modelo que mejor se ajusta es el de red neuronal simple, ya que mantiene un error absoluto, una varianza y un sesgo relativamente bajos, sin caer en underfitting u overfitting.

---

## **Diagnóstico y explicación de los resultados**

El modelo de una red neuronal simple fue el que mejor resultado dio: un error de 120,848 dólares, un sesgo de 3.9e10 y una varianza de 7.4e8. A pesar de que estos valores pueden parecer altos (influenciado en gran medida por el rango de valores en los precios), el resultado es bueno y podemos observar en la gráfica que se tiene un buen modelo. En el futuro, mejoras posibles pueden encontrarse en el número de épocas de entrenamiento, ajuste de hiperparámetros diferentes (como tasa de aprendizaje o porcentaje de validación) y  modificaciones en las capas utilizadas para las redes.

![image](https://user-images.githubusercontent.com/67491368/190276253-887bafcc-06db-40bc-8578-140a3f913417.png)

