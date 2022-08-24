# **Implementación de una técnica de aprendizaje máquina sin el uso de un framework.**

_Myroslava Sánchez Andrade_

En este repositorio se implementó un modelo de regresión multilineal para la predicción del precio de un diamante.

## Extract

De la plataforma Kaggle, se descargó un [dataset](https://www.kaggle.com/datasets/shivam2503/diamonds) que contiene los precios y otros atributos de casi 54,000 diamantes. Está compuesto por las variables: carat, cut, color, clarity, x (longitud), y (ancho), z (profundidad), depth, table y precio.

Para verificar y analizar la estructura del dataset, utilicé la función `create_report()` de la librería dataprep.eda que crea un reporte estadísitco descriptivo y general.

En este reporte se observó:

![image](https://user-images.githubusercontent.com/67491368/186533179-f0bc7fa0-6309-4468-9a7c-dc9b9601ddd2.png)
- No se cuenta con ninguna celda vacía en el DataFrame, por lo que no fue necesario eliminar columnas por tener valores faltantes.

![image](https://user-images.githubusercontent.com/67491368/186533368-7600fa53-3400-4bc5-824a-93d23204d127.png)
- Podemos observar que las variables x, y, z y quilates tienen una correlación mayor a 0.94 entre ellas, lo que nos permitió eliminar a 3 de estas. Se decidió mantener la variable de quilates pues es la que mayor correlación tiene con el precio (output).

## Transform

Después de analizar cada variable se realizaron las siguientes transformaciones:

- Se estandarizó la escala del porcentaje de las variables depth y table. En lugar de una escala 0-100 se estandarizó a una escala 0-1.

- Conversión de las variables categóricas cut, color y clarity, a variables numéricas. Fue posible hacer esta conversión debido a que las variables tienen un orden (worst to best)

## Load

Se exportó el DataDrame resultante a un csv.

## Modeling (multi linear regression)

Se dividió el DataFrame con la data limpia en las variables de entrada y la de salida. A su vez se dividieron estas para el entrenamiento (80%) y el training (20%).

Se utilizó el algoritmo de Gradient Descent como algoritmo de optimización. Este consiste en encontrar el minimo local de una función diferenciable (diferencia del valor predecido menos el valor real dadas las betas => slope). En cada iteración (época) se busca reducir el error y así encontrar los pesos de las variables (betas 1 -n) y el bias (beta 0) más óptimo posible.

El entrenamiento tenía un total de 1000 épocas y un learning rate de 0.05. Después de haber probado mi modelo en muchas ocasiones, a veces obtenía resultados muy buenos, y a veces muy malos. Por lo que decidí implementar el entrenamiento de 1000 épocas un total de 500 veces y tomar el mejor resultado (el de menor error).

## Conclusiones

Debido a que se contaban con un total de 6 variables independientes, el resultado del entrenamiento no parecía siempre el más óptimo. Se podría aplicar un algoritmo genético para reducir el espacio de búsqueda (creación de valores aleatorios en cada iteración).
