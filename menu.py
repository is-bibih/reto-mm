import reto

def menu():
    max_choice = 2
    menu_str = 'Reto de métodos matemáticos\n' \
        + '1) identificar una canción\n' \
        + '2) crear una nueva base de datos\n' \
        + '0) salir\n\n' \
        + 'Escribe un número para elegir la opción correspondiente.'
    print(menu_str)
    ans = input()
    # validar entrada
    try:
        ans = int(ans)
        if ans > max_choice or ans < 0:
            print('Respuesta inválida.\n')
            menu()
    except:
        print('Respuesta inválida.\n')
        menu()
    # ver casos
    if ans == 0:
        print('bai')
    elif ans == 1: # identificar cancion
        song_path = input('\nIntroduce la ruta al archivo .wav\n')
        print('Procesando archivo...')
        features = reto.get_feature_vector(song_path)
        print('Archivo procesado.')
        db_path = input('\nIntroduce la ruta al achivo .npz de la base de datos.\n')
        print('Cargando base de datos...')
        db = reto.load_db(db_path)
        print('Comenzando comparación con base de datos...')
        results = reto.compare_all(features, db) 
        match, score = reto.get_best_match(results)
        print('La mejor coincidencia fue {} (diferencia de {})'.format(match, score))
        print('\n\n')
        menu()
    elif ans == 2: # hacer base de datos
        db_dir = input('\nIntroduce la ruta al directorio ' \
                       'con los archivos para crear la base de datos.\n')
        outfile = input('\nIntroduce un nombre de archivo para almacenar la base de datos.\n')
        print('Generando base de datos...')
        reto.generate_db(db_dir, outfile)
        print('Base de datos generada en {}'.format(outfile))
        print('\n\n')
        menu()

menu()
