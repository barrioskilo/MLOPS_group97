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
print("\nDatos con etiqueta 1:")
print(data_tensor[indice_clase_1])

# Verificar si hay alguna instancia con etiqueta 2
indices_clase_2 = (labels_tensor == 2).nonzero(as_tuple=True)[0]

if len(indices_clase_2) > 0:
    # Imprimir el tensor correspondiente a la primera instancia con etiqueta 2
    indice_clase_2 = indices_clase_2[0]
    print("\nDatos con etiqueta 2:")
    print(data_tensor[indice_clase_2])
else:
    print("\nNo hay instancias con etiqueta 2 en el conjunto de datos.")

# Imprimir un tensor con algún valor positivo
indice_valor_positivo = (data_tensor > 0).nonzero(as_tuple=True)
if len(indice_valor_positivo[0]) > 0:
    # Imprimir el tensor con algún valor positivo
    print("\nTensor con algún valor positivo:")
    print(data_tensor[indice_valor_positivo])
else:
    print("\nNo hay valores positivos en el conjunto de datos.")

# Get the shape of one image
shape_of_one_image = data_tensor[0].shape

# Print the shape
print("\nShape of one image:", shape_of_one_image)

