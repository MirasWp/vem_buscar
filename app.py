from flask import Flask, render_template, request, jsonify
import face_recognition
import numpy as np
import base64
import cv2
import pyodbc  

app = Flask(__name__)

# Configurações do banco de dados
CONN_STR = (
    "DRIVER={ODBC Driver 18 for SQL Server};"
    "SERVER=LUCAS;"        # ou "localhost\\SQLEXPRESS"
    "DATABASE=APTO_TESTE;"
    "Trusted_Connection=yes;"      # indica autenticação do Windows
    "TrustServerCertificate=yes;"  # evita problemas de certificado SSL
    "Encrypt=yes;"
)

DIST_THRESHOLD = 0.6  # distância máxima para considerar correspondência

# Carrega embeddings do banco de dados usando CNN
def carregar_embeddings():
    conn = pyodbc.connect(CONN_STR)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT e.ID_ALUNO, a.NOME, e.EMBEDDING
        FROM EXTRACAO_FACIAL e
        JOIN ALUNOS a ON a.ID_ALUNO = e.ID_ALUNO
    """)
    resultados = cursor.fetchall()
    conn.close()

    embeddings = [(row.ID_ALUNO, row.NOME, np.frombuffer(row.EMBEDDING, dtype=np.float64)) 
                  for row in resultados]
    return embeddings

# Carrega todos os embeddings na inicialização
EMBEDDINGS = carregar_embeddings()

# Rota principal
@app.route('/')
def index():
    return render_template('index.html')

# Rota para reconhecimento facial
@app.route('/reconhecer', methods=['POST'])
def reconhecer():
    data = request.get_json()
    image_data = data['image'].split(',')[1]  # remove "data:image/jpeg;base64,"
    img_bytes = base64.b64decode(image_data)
    np_arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Detecta rostos
    face_locations = face_recognition.face_locations(rgb_img)
    
    # Extrai embeddings usando CNN
    face_encodings = face_recognition.face_encodings(rgb_img, face_locations, model="cnn")

    resultados = []

    for encoding in face_encodings:
        distancias = [np.linalg.norm(encoding - e[2]) for e in EMBEDDINGS]
        if distancias:
            min_dist = min(distancias)
            index = distancias.index(min_dist)
            if min_dist < DIST_THRESHOLD:
                id_aluno, nome, _ = EMBEDDINGS[index]
                resultados.append({'id': id_aluno, 'nome': nome, 'distancia': float(min_dist)})

    return jsonify(resultados)

if __name__ == '__main__':
    app.run(debug=True)