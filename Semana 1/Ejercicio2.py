import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# 1. Definimos el modelo (3 entradas: [Nivel de cansancio, Interés en película, Clima])
model = keras.Sequential([
    layers.Dense(1, input_shape=(3,), activation='sigmoid', use_bias=True)
])

# 2. Establecer criterios de importancia (Pesos y Sesgo)
# Entradas: [Cansancio (0=Fresco, 1=Agotado), Calificación Película (0-1), Buen Clima (0=Lluvia, 1=Soleado)]
# Pesos:
#   Cansancio: -7.0 (Si estás cansado, resta muchos puntos a la idea de salir)
#   Interés: 9.0    (Si la película es un estreno increíble, suma mucho)
#   Clima: 4.0      (Si hace buen tiempo, ayuda a salir)
pesos = np.array([[-7.0], [9.0], [4.0]]) 

# Sesgo: -2.0 
# (Es un pequeño "empujón" hacia quedarse en casa para proteger el presupuesto de $1.35M)
sesgo = np.array([-2.0]) 

model.layers[0].set_weights([pesos, sesgo])

# 3. Escenario de prueba:
# Estás algo cansado (0.7), la película tiene un 9/10 (0.9) y está lloviendo un poco (0.3)
entrada_hoy = np.array([[0.7, 0.9, 0.3]])

# 4. Predicción
salida = model.predict(entrada_hoy)
probabilidad = salida[0][0]

# 5. Resultado
print(f"\n--- Análisis de Entretenimiento ---")
print(f"Probabilidad de ir al cine: {probabilidad:.4f}")

if probabilidad >= 0.5:
    print("🎬 ¡A las salas! La película vale el esfuerzo y el gasto.")
else:
    print("🏠 Quédate en casa. Ahorra esos pesos y descansa un poco.")