import face_recognition
import numpy as np
import os
from os.path import join, isfile, isdir
import pyodbc

# Diretório com as fotos já recortadas em 180x180
DIR_FACES = r"C:\Users\lucas\OneDrive\Desktop\ATP\Faces"

# Modo do detector: "hog" (rápido, CPU) ou "cnn" (mais preciso, pesado)
MODO_FACE = "hog"

# String de conexão usando autenticação do Windows
CONN_STR = (
    "DRIVER={ODBC Driver 18 for SQL Server};"
    "SERVER=LUCAS;"        # ou "localhost\\SQLEXPRESS"
    "DATABASE=APTO_TESTE;"
    "Trusted_Connection=yes;"      # indica autenticação do Windows
    "TrustServerCertificate=yes;"  # evita problemas de certificado SSL
    "Encrypt=yes;"
)

def get_connection():
    """Cria uma conexão com o SQL Server"""
    return pyodbc.connect(CONN_STR)

def salvar_embedding(id_aluno, embedding, conn):
    """Salva um vetor facial (embedding) no banco"""
    cursor = conn.cursor()
    embedding_bytes = embedding.tobytes()
    cursor.execute("""
        INSERT INTO EXTRACAO_FACIAL (ID_ALUNO, EMBEDDING)
        VALUES (?, ?)
    """, (id_aluno, embedding_bytes))
    conn.commit()
    print(f"[OK] Embedding salvo para aluno ID {id_aluno}")

def get_id_aluno(nome_aluno, conn):
    """Busca ID_ALUNO pelo nome da pasta (nome do aluno). Se não existir, insere novo aluno."""
    if not nome_aluno or nome_aluno.strip() == "":
        raise ValueError("O nome do aluno não pode ser vazio")

    nome_aluno = nome_aluno.strip()  # remove espaços extras
    cursor = conn.cursor()

    # Verifica se já existe (case-insensitive)
    cursor.execute("SELECT ID_ALUNO FROM ALUNOS WHERE UPPER(NOME) = UPPER(?)", (nome_aluno,))
    row = cursor.fetchone()

    if row:
        return row[0]
    else:
        # Insere novo aluno
        cursor.execute(
            "INSERT INTO ALUNOS (NOME) OUTPUT INSERTED.ID_ALUNO VALUES (?)",
            (nome_aluno,)
        )
        id_aluno = cursor.fetchone()[0]
        conn.commit()
        return id_aluno

def processar_faces(diretorio_faces):
    """Processa todas as imagens do diretório e salva embeddings no banco"""
    with get_connection() as conn:
        for aluno in os.listdir(diretorio_faces):
            pasta_aluno = join(diretorio_faces, aluno)
            if not isdir(pasta_aluno):
                continue

            print(f"\n[ALUNO] {aluno}")
            try:
                id_aluno = get_id_aluno(aluno, conn)
            except ValueError as e:
                print(f"  [ERRO] {e}")
                continue

            for filename in os.listdir(pasta_aluno):
                path = join(pasta_aluno, filename)
                if not isfile(path) or not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue

                print(f"  [PROCESSANDO] {filename}")
                imagem = face_recognition.load_image_file(path)

                # HOG detector (rápido, roda 100% em CPU)
                boxes = face_recognition.face_locations(imagem, model=MODO_FACE)
                embeddings = face_recognition.face_encodings(imagem, boxes)

                if embeddings:
                    vetor = np.array(embeddings[0], dtype=np.float64)
                    salvar_embedding(id_aluno, vetor, conn)
                else:
                    print(f"  [ERRO] Nenhum rosto encontrado em: {filename}")

def carregar_embeddings():
    """Lê todos os embeddings do banco e reconverte para numpy"""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT ID_ALUNO, EMBEDDING FROM EXTRACAO_FACIAL")
        resultados = cursor.fetchall()

    embeddings = [(id_aluno, np.frombuffer(embedding_bytes, dtype=np.float64))
                  for id_aluno, embedding_bytes in resultados]
    return embeddings

if __name__ == "__main__":
    processar_faces(DIR_FACES)

    # Teste: carregar de volta do banco
    emb = carregar_embeddings()
    print(f"\n[TESTE] Carregados {len(emb)} embeddings do banco.")
    if emb:
        print(f"Exemplo: ID_ALUNO={emb[0][0]}")
        print(emb[0][1])
