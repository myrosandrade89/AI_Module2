# **Título**

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
