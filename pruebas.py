import os
import reto
import glob
import numpy as np
import pandas as pd
from timeit import timeit 

# probar carga de cancion y sfft
# probar para muchas canciones
def test_load(file_list):
    results = np.zeros(len(file_list))
    for i, file in enumerate(file_list):
        func = lambda: reto.get_sfft(file)
        results[i] = timeit(func, number=1)
    return results

# probar sacar matrices para cancion
# probar para muchas canciones
def test_matrices(file_list):
    results = np.zeros(len(file_list))
    for i, file in enumerate(file_list):
        x, freqs = reto.get_sfft(file)
        func = lambda: reto.make_new_bins(x, freqs)
        results[i] = timeit(func, number=25)
    return results

# probar cargar base de datos
def test_load_db(db_path):
    func = lambda: reto.load_db(db_path)
    return timeit(func, number=25)

# probar comparar con base de datos
# probar para muchas canciones
def test_compare(db):
    max_n = len(db)
    # matriz: [cantidad_comparada x cancion]
    results = np.zeros((max_n, max_n))
    ran = np.random.default_rng()
    for n in range(max_n):
        for i, features in enumerate(db.values()):
            keys = ran.choice(list(db.keys()), n+1)
            db_small = {key:db[key] for key in keys}
            func = lambda: reto.compare_all(features, db_small)
            results[n, i] = timeit(func, number=25)
    return results

# probar resultados de cada cancion
def test_predict(new_db, ref_db, should_stop_early=True):
    # matriz: [cantidad_canciones x 3]
    # columnas: original, encontrada, puntuacion
    results = [0] * len(new_db)
    i = 0
    for song, features in new_db.items():
        print('comparing {} of {}'.format(i+1, len(new_db)))
        scores = reto.compare_all(features, ref_db,
                                  should_stop_early=should_stop_early)
        best, best_score = reto.get_best_match(scores)
        results[i] = [song, best, best_score]
        i += 1
    return results

# probar con canciones mezcladas
def test_mixed(db):
    # para cada cancion, cambiar la mitad de la matriz
    #   con la de otra
    mixing = {}
    new_db = {}
    for song, features in db.items():
        features = np.array(features)
        # random generator
        ran = np.random.default_rng()
        # seleccionar cancion al azar para combinar
        rand_song = ran.choice(list(db.keys()))
        # registrar
        mixing[song] = rand_song
        # matriz de cancion al azar
        rand_feat = db[rand_song]
        # ver la cantidad de columnas de original
        n_cols = np.shape(features)[1]
        # cantidad de columnas de la otra
        n_cols_r = np.shape(rand_feat)[1]
        # indices de las columnas de la matriz
        idx = np.arange(min(n_cols, n_cols_r))
        # muestra aleatoria de la mitad de los idx
        idx = ran.choice(idx, int(n_cols/2))
        # combinar
        features[:, idx] = rand_feat[:, idx]
        new_db[song] = features
    # comparar con base de datos
    results = test_predict(new_db, db, should_stop_early=False)
    # columnas: cancion original, cancion al azar,
    #   mejor coincidencia, puntuacion
    results = [[song, mixing[song], best, best_score]
                for (song, best, best_score) in results]
    return results

# probar con orden distinto

# probar con muestras pequeñas
# al principio, medio y final
def test_short(db):
    start_db = {}
    middle_db = {}
    end_db = {}
    results = []
    for song, features in db.items():
        n_cols = np.shape(features)[1]
        half = int(n_cols/2)
        new_length = int(n_cols/5)
        start_db[song] = features[:, :new_length]
        middle_db[song] = features[:, half:half+new_length]
        end_db[song] = features[:, -new_length:]
    for short_db in (start_db, middle_db, end_db):
        results.append(test_predict(short_db, db))
    return results

# probar con ruido
# probar con tres niveles
def test_noisy(db):
    noisy_db = {}
    ran = np.random.default_rng()
    for song, features in db.items():
        mean_freq = np.mean(features)
        spread = mean_freq
        noise = ran.normal(scale=spread, size=np.shape(features))
        noisy_db[song] = np.abs(features + noise)
    return test_predict(noisy_db, db)


# hacerlo
files = glob.glob(os.path.join('db', '*.wav'))
db_dict = dict(reto.load_db('db.npz'))
del db_dict['./db/music_zapsplat_staring_through_midday.wav']
del db_dict['./db/audio_hero_Black-Fedora_SIPML_J-0310.wav']
del db_dict['./db/music_david_gwyn_jones_cheesy_christmas.wav']
test_results = np.array(test_noisy(db_dict), dtype=object)
print(test_results)
np.savetxt('pruebas/noisy_sin_cheesy.csv',
           test_results,
           fmt=['%s', '%s', '%d'],
           delimiter=',',
           header='cancion,coincidencia,puntaje')
#for res, name in zip(test_results, ('principio', 'mitad', 'final')):
#    filename = 'pruebas/corto_{}.csv'.format(name)
#    res = np.array(res, dtype=object)
#    np.savetxt(filename,
#               res,
#               fmt=['%s', '%s', '%d'],
#               delimiter=',',
#               header='cancion,coincidencia,puntaje')
#df = pd.DataFrame(test_results, columns=["Cancion", "Coincidencia", "Puntaje"], dtype="string")
#df.to_csv('pruebas/comparar.csv')
#del db_dict['./db/music_zapsplat_staring_through_midday.wav']

