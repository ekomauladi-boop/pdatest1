import os
import pandas as pd
import streamlit as st
from langchain_community.document_loaders import DataFrameLoader
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# Konfigurasi Halaman Streamlit
st.set_page_config(
    page_title="Chatbot Cagar Budaya Masjid Syuhada",
    page_icon="🕌",
    layout="wide"
)

st.title("🕌 Chatbot Informasi Cagar Budaya Masjid Syuhada")
st.caption("Aplikasi RAG berbasis Metadata DCMI menggunakan Google Gemini")

# Ambil Google Gemini API Key dari Streamlit Secrets
google_api_key = st.secrets.get("GOOGLE_API_KEY") or os.environ.get("GOOGLE_API_KEY")

if not google_api_key:
    st.error("⚠️ API Key Google Gemini belum ditemukan. Tambahkan `GOOGLE_API_KEY` di menu Secrets Streamlit Cloud.")
    st.stop()

# Fungsi Inisialisasi Data & Vector Store
@st.cache_resource(show_spinner="Memuat data metadata cagar budaya...")
def init_vector_store():
    file_path = "Metadata_DCMI_Masjid_Syuhada.xlsx"
    
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    else:
        df = pd.read_excel(file_path)
    
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
    
    loader = DataFrameLoader(df, page_content_column="page_content")
    documents = loader.load()
    
    # Menggunakan model embedding bawaan yang stabil
    embeddings = GoogleGenerativeAIEmbeddings(
        model="text-embedding-004", 
        google_api_key=google_api_key
    )
    
    vectorstore = Chroma.from_documents(documents=documents, embedding=embeddings)
    return vectorstore

try:
    vectorstore = init_vector_store()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
except Exception as e:
    st.error(f"Gagal memuat sistem: {e}")
    st.stop()

# Prompt & RAG Chain
system_prompt = (
    "Anda adalah asisten AI ahli Cagar Budaya dan Arsitektur Masjid Syuhada Yogyakarta.\n"
    "Gunakan konteks informasi berikut untuk menjawab pertanyaan pengguna dengan presisi.\n"
    "Jika data tidak tersedia, katakan jujur bahwa informasi belum tersedia.\n\n"
    "Konteks Data:\n"
    "{context}\n"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash", 
    temperature=0.2, 
    google_api_key=google_api_key
)

def format_docs(docs):
    return "\n\n---\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "input": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Riwayat Chat
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Halo! Silakan tanyakan apa saja seputar elemen bangunan Masjid Syuhada."}
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input Chat
if user_input := st.chat_input("Ketik pertanyaan di sini..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Berpikir..."):
            try:
                response = rag_chain.invoke(user_input)
                st.markdown(response)
            except Exception as err:
                response = f"Terjadi kendala: {err}"
                st.error(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})
