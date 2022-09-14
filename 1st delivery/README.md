# **Implementación de una técnica de aprendizaje máquina sin el uso de un framework.**

_Autor: Myroslava Sánchez Andrade_
<br>_Fecha de creación: 31/08/2022_
<br>_Última modificación: 13/09/2022_

---

## **Visión general**

El propósito de este repositorio es realizar la programación de un algoritmo de predicción sin el uso de frameworks.
<br>Las librerías necesarias para correr el código son: pandas, numpy, matplotlib y dataprep.

Video de Google Colab: https://youtu.be/x_i4Z11iG4Y

---

## **Extract**

De la plataforma Kaggle, se descargó un [dataset](https://www.kaggle.com/datasets/shivam2503/diamonds) que contiene los precios y otros atributos de casi 54,000 diamantes. Está compuesto por las siguientes variables:

- **_Independent variables:_** carat, cut, color, clarity, x (longitud), y (ancho), z (profundidad), depth, table. Siendo categóricas las variables: `['cut', 'color', 'clarity']`.
- **_Dependent variable:_** price.

Para el anális de la estructura del dataset se utilizó la función **[create_report()](https://docs.dataprep.ai/user_guide/eda/create_report.html)** de la librería dataprep.eda que crea un reporte estadísitco descriptivo.

En este reporte se observó:
![image](https://user-images.githubusercontent.com/67491368/186533179-f0bc7fa0-6309-4468-9a7c-dc9b9601ddd2.png)

- No se cuenta con ninguna celda vacía en el DataFrame, por lo que no fue necesario eliminar columnas por tener valores faltantes; también pudimos identificar que se cuenta con 3 variables categóricas.

![image](https://user-images.githubusercontent.com/67491368/186533368-7600fa53-3400-4bc5-824a-93d23204d127.png)

- Podemos observar que las variables x, y, z y quilates tienen una correlación mayor a 0.94 entre ellas.

- Los rangos de las variables (valores máximos y mínimos).

---

## **Transform**

Después de analizar cada variable, su estructura y contenido, se realizaron las siguientes transformaciones:

- Se eliminaron 3 de 4 columnas que tenían una relación mayor a 0.95; se decidió mantener la variable de quilates pues es la que mayor correlación tiene con el precio (output)

- Se estandarizó la escala del porcentaje de las variables depth y table. En lugar de una escala 0-100 se estandarizó a una escala 0-1.

- Conversión de las variables categóricas cut, color y clarity, a variables numéricas. Fue posible hacer esta conversión debido a que las variables tienen un orden (worst to best)

---

## **Load**

Se exportó el DataDrame resultante a un csv.

---

## **Modeling (multi linear regression)**

Se dividió el DataFrame con la data limpia en las variables de entrada y la de salida. A su vez se dividieron estas para el entrenamiento (80%) y el training (20%).

Se utilizó el algoritmo de Gradient Descent como algoritmo de optimización. Este consiste en encontrar el minimo local de una función diferenciable (diferencia del valor predecido menos el valor real dadas las betas => slope). En cada iteración (época) se busca reducir el error y así encontrar los pesos de las variables (betas 1-n) y el bias (beta 0) más óptimo posible.

El entrenamiento tenía un total de 1000 épocas y un learning rate de 0.05. Después de haber probado mi modelo en muchas ocasiones, a veces obtenía resultados muy buenos, y a veces muy malos. Por lo que decidí implementar el entrenamiento de 1000 épocas un total de 30 veces y tomar el mejor resultado (el de menor error).

---

## **Conclusiones**

A pesar de que se contaba con un total de 6 variables independientes, el resultado del entrenamiento fue bastante bueno, al igual que el resultado del test. <br>Para optimizar este modelo se podría aplicar un algoritmo genético para reducir el espacio de búsqueda (número de valores aleatorios generados antes de optimizar el error).
