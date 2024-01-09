import torch

# Cargar el archivo
loaded_data = torch.load('/Users/anderbarriocampos/Desktop/MLOPS_group97/data/processed/processed_data.pt')

# Acceder a los datos y etiquetas
data_tensor = loaded_data['data']
labels_tensor = loaded_data['labels']

# Imprimir las primeras 5 filas de los tensores
print("Primeras 5 filas de Datos:")
print(data_tensor[:5])

print("\nPrimeras 5 filas de Etiquetas:")
print(labels_tensor[:5])

# Buscar el primer índice donde la etiqueta sea igual a 1
indice_clase_1 = (labels_tensor == 1).nonzero(as_tuple=True)[0][0]

# Imprimir el tensor correspondiente a la primera instancia con etiqueta 1
print("Datos con etiqueta 1:")
print(data_tensor[indice_clase_1])
