import numpy as np
import os
import glob
from librosa import display
import librosa

# ----- parametros para leer archivos de musica -----

# cantidad de muestras por segundo
sample_rate = 11025

# ---------------------------------------------------


# ----- crear rangos para clasificar frecuencias -----

# cantidad de rangos
n_bins = 128
# frecuencia maxima y minima log2(herz)
# corresponden a las de un piano estandar
log_min = np.log2(1)
log_max = np.log2(4096)
# generar rangos espaciados logaritmicamente
bin_limits = np.logspace(log_min, log_max, num=n_bins, base=2.0)
# quitar el limite inferior del primer rango
bin_limits = bin_limits[1:]

# ---------------------------------------------------


# --------------- parametros para fft ---------------

# cantidad de muestras para calcular un "periodo"
framelength = 4096
# cantidad de muestras a avanzar en cada iteracion
# (es menor a framelength entonces hay sobrelape)
superpose = int(2*framelength/3) # (2/3)

# ---------------------------------------------------

def get_sfft(filename, sample_rate=sample_rate, duration=None,
             n_fft=framelength, hop_length=superpose):
    """
    Carga un archivo de audio, hace el SFFT y obtiene las
    frecuencias correspondientes.
    """
    # cargar el audio con la frecuencia de muestreo por la duracion dada
    song, sr = librosa.load(filename, sr=sample_rate, duration=duration)
    # hacer el sfft
    # (la matriz tiene forma n_frecuencias_distintas x n_tiempos)
    # (cada elemento es la amplitud de esa frecuencia en ese tiempo)
    x = librosa.stft(song, n_fft=framelength, hop_length=superpose)
    # obtiene las frecuencias que se usaron para el sfft
    freqs = librosa.fft_frequencies(sr=sr, n_fft=framelength)
    return x, freqs

def make_new_bins(x, x_bins, new_bins=bin_limits):
    """
    Regresa una matriz con las frecuencias de mayor amplitud
    dentro de cada rango, para cada tiempo.

    La matriz tiene forma len(new_bins) x n_cols
    """
    # almacenar la cantidad de columas de la matriz dada
    _, n_cols = np.shape(x)
    # ver a cuales rangos pertenece cada frecuencia
    x_classes = np.digitize(x_bins, new_bins)
    # crear la matriz para almacenar las frecuencias maximas
    new_x = np.zeros((len(new_bins), n_cols))
    # encontrar la amplitud maxima dentro de cada rango
    # (itera sobre cada rango)
    for i in range(len(new_bins)):
        # obtiene el indice del primer elemento
        #   que pertenece al rango
        first = np.argmax(x_classes == i)
        # los indices de todos los elementos que pertenecen
        #   al rango
        idx = x_classes == i
        # si hay elementos en el rango
        # si no, se queda en 0
        if any(idx):
            # indice del maximo por columna (por tiempo)
            #   de los elementos en el rango
            max_idx = np.argmax(x[idx], axis=0)
            # obtiene las frecuencias que corresponden
            #   a los maximos
            new_x[i] = x_bins[max_idx + first]
    return new_x

def get_feature_matrix(filename, sample_rate=sample_rate,
                       duration=None, n_fft=framelength,
                       hop_length=superpose, ranges=bin_limits):
    """
    Regresa la matriz de frecuencias dado un archivo de audio.
    """
    x, freqs = get_sfft(filename, sample_rate=sample_rate, duration=duration,
                        n_fft=framelength, hop_length=superpose)
    features = make_new_bins(x, freqs)
    return np.array(features)

def load_db(path):
    """
    Regresa un diccionario con la base de datos.
    """
    return np.load(path)

def get_difference(v1, v2):
    """
    Regresa la medida de diferencia de dos matrices (norma L1).
    """
    # suma del valor absoluto de las diferencias entre elementos
    dif = np.sum(np.abs(v1 - v2))
    return dif

# define convolution
def convo(f, v1, v2, window=None, step=None, should_stop_early=True):
    """
    "Barre" las matrices para evaluar la función de diferencia
    sobre ellas.
    """
    # cantidad de tiempos en la matriz v1 (la muestra)
    _, n_t1 = np.shape(v1)
    # cantidad de tiempos en la matriz v2 (de la base de datos)
    _, n_t2 = np.shape(v2)

    # definir parametros

    # default
    if not window:
        window = int(n_t1 / 20)
    # si es una fraccion
    elif window < 1:
        window = int(n_t1 * window)
    # si es un entero entonces se usa eso

    # default
    if not step:
        step = int(window * 2/3)
    # si es una fraccion
    elif step < 1:
        step = int(window * step)
    # si es un entero usa eso

    i = 0           # indice para v1
    value = 0       # para acumular las medidas de diferencia
    same_count = 0  # para contar cuantas veces consecutivas es igual

    # iterar mientras no llegue al final y haya sido igual
    #   menos de 5 veces consecutivas
    while (window + i*step < n_t1) and (same_count < 5):

        # obtener columnas correspondientes al pedazo actual
        # (es un rango de tiempo)
        v1_i = v1[:, i*step : window + i*step]
        # initialize counter for the other matrix
        j = 0       # indice para v2

        while (window + j*step < n_t2) and (same_count < 5):
            v2_j = v2[:, j*step : window + j*step]
            # incrementar el indice de v2
            j += 1
            # obtener la medida de diferencia
            dif = f(v1_i, v2_j)

            # si es completamente igual
            if dif == 0:
                # pasar al pedazo que sigue
                i += 1
                v1_i = v1[:, i*step : window + i*step]
                # si debe detenerse cuando sea
                #   igual varias veces consecutivas
                if should_stop_early:
                    # incrementar las veces que ha sido igual
                    same_count += 1
            # si es distinto
            else:
                # reiniciar el contador de veces consecutivas
                same_count = 0
                # sumar diferencia de este pedazo a total
                value += dif
        # pasar al siguiente pedazo
        i += 1

    # ver si se detuvo porque era igual
    stopped_early = same_count > 4
    return value, stopped_early

def compare_all(sample, db, should_stop_early=True):
    """
    Regresa un diccionario con la diferencia respecto a
    cada elemento en la base de datos.
    """
    # inicializa el diccionario
    scores = {}
    # calcula la diferencia para cada elemento
    for file, features in db.items():
        score, _ = convo(get_difference,
                         sample,
                         features,
                         should_stop_early=should_stop_early)
        scores[file] = score
    return scores

def get_best_match(scores_dict):
    """
    Regresa la mejor coincidencia dado el diccionario
    de diferencias.
    """
    # indice del mejor
    idx = np.argmin(list(scores_dict.values()))
    # clave del mejor
    key = list(scores_dict.keys())[idx]
    return key, scores_dict[key]

def generate_db(dir_path, out_path):
    """
    Genera una base de datos con los archivos .wav
    en un directorio.
    """
    # obtiene una lista de los wav en el directorio
    files = glob.glob(os.path.join(dir_path, '*.wav'))
    # lista para almacenar las matrices de cada archivo
    features = [0] * len(files)
    # iterar sobre los archivos
    for i in range(len(files)):
        print('Procesando {}/{}: {}...'.format(i + 1, len(files), files[i]))
        # calcular la matriz
        features[i] = get_feature_matrix(files[i])
    # diccionario archivo: matriz
    db_dict = dict(zip(files, features))
    # guardar las entradas del diccionario a un archivo .npz
    np.savez(out_path, **db_dict)

