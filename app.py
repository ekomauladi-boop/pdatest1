import os
import pandas as pd
import streamlit as st
from langchain_community.document_loaders import DataFrameLoader
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# 1. Konfigurasi Halaman Streamlit
st.set_page_config(
    page_title="Chatbot Cagar Budaya Masjid Syuhada (Gemini)",
    page_icon="🕌",
    layout="wide"
)

st.title("🕌 Chatbot Informasi Cagar Budaya Masjid Syuhada")
st.caption("Aplikasi RAG berbasis Metadata DCMI menggunakan Google Gemini")

# 2. Ambil Google Gemini API Key dari Streamlit Secrets / Environment
google_api_key = st.secrets.get("GOOGLE_API_KEY") or os.environ.get("GOOGLE_API_KEY")

if not google_api_key:
    st.error("⚠️ API Key Google Gemini belum ditemukan. Silakan tambahkan `GOOGLE_API_KEY` di `.streamlit/secrets.toml` atau di Secrets Streamlit Cloud.")
    st.stop()

# 3. Fungsi Ingest Data & Inisialisasi Vector Store (Caching)
@st.cache_resource(show_spinner="Memuat dan mengindeks data metadata dengan Gemini...")
def init_vector_store():
    # Membaca file Excel
    file_path = "Metadata_DCMI_Masjid_Syuhada.xlsx"
    df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)
    
    # Gabungkan isi seluruh kolom ke dalam satu teks kaya konteks per baris
    def format_row(row):
        return (
            f"ID Elemen: {row.get('Element_ID', '')}\n"
            f"Nama Elemen: {row.get('Title_Nama_Elemen', '')}\n"
            f"Kategori: {row.get('Category_Elemen', '')}\n"
            f"Lokasi Ruang: {row.get('Location_Ruang', '')}\n"
            f"Massa Bangunan: {row.get('Massa_Bangunan', '')}\n"
            f"Material & Finishing: {row.get('Material_Finishing', '')}\n"
            f"Deskripsi Fisik: {row.get('Physical_Description', '')}\n"
            f"Tingkat Signifikansi: {row.get('Significance_Level', '')}\n"
            f"Kondisi Fisik: {row.get('Condition_Status', '')}\n"
            f"Keaslian Sejarah: {row.get('Historical_Authenticity', '')}\n"
            f"Rekomendasi Pelestarian: {row.get('Preservation_Recommendation', '')}\n"
            f"Dokumen Rujukan: {row.get('Source_Document', '')}"
        )
    
    df['page_content'] = df.apply(format_row, axis=1)
    
    # Load ke LangChain Document
    loader = DataFrameLoader(df, page_content_column="page_content")
    documents = loader.load()
    
    # Inisialisasi Embeddings & Vector Store (ChromaDB) menggunakan Google GenAI
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)
    vectorstore = Chroma.from_documents(documents=documents, embedding=embeddings)
    return vectorstore

# Inisialisasi Vector Store
vectorstore = init_vector_store()
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 4. Prompt Template RAG
system_prompt = (
    "Anda adalah asisten AI ahli Cagar Budaya dan Arsitektur Masjid Syuhada Yogyakarta.\n"
    "Gunakan potongan konteks informasi DCMI berikut untuk menjawab pertanyaan pengguna.\n"
    "Jika Anda tidak mengetahui jawabannya berdasarkan konteks yang diberikan, katakan secara jujur "
    "bahwa informasi tersebut belum tersedia dalam basis data cagar budaya.\n\n"
    "Konteks Data:\n"
    "{context}\n"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

# 5. Inisialisasi LLM Google Gemini & Chain
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2, google_api_key=google_api_key)

def format_docs(docs):
    return "\n\n---\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "input": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 6. Session State untuk Riwayat Chat
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Halo! Saya asisten virtual Cagar Budaya Masjid Syuhada berbasis Gemini. Ada yang bisa saya bantu mengenai elemen arsitektur, kondisi fisik, atau sejarah bangunan?"}
    ]

# Tampilkan Riwayat Chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 7. Input Pengguna
if user_input := st.chat_input("Tanyakan tentang elemen bangunan Masjid Syuhada (misal: kondisi kubah, tegel teraso, dll)..."):
    # Tampilkan pesan user
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Respon AI
    with st.chat_message("assistant"):
        with st.spinner("Mencari data metadata dengan Gemini..."):
            response = rag_chain.invoke(user_input)
            st.markdown(response)
    
    # Simpan respon AI ke riwayat
    st.session_state.messages.append({"role": "assistant", "content": response})
