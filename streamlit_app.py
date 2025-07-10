import os
import torch
import streamlit as st
import pinecone
from langchain_mistralai import ChatMistralAI
from langchain.prompts.chat import ChatPromptTemplate
from langchain.chains import RetrievalQA
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables import RunnableLambda
from langchain.prompts import PromptTemplate

# Set up environment variables
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = "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"
os.environ["LANGCHAIN_PROJECT"] = "pr-long-replacement-23"
os.environ["MISTRAL_API_KEY"] = "OWEMpjjsDfW4CEJ5gl1ZC52gjBOqFOu5"
os.environ["PINECONE_API_KEY"] = "pcsk_3d9skL_3zcxDDceuDNr2kpiEvyTSi51NLkanLw2SsrFivW57pnJjqihy5K8csZj4X8svxa"

# Initialize the LLM (Language Model) with the system prompt in Serbian
system_prompt = """
Dobrodošli u Paragraf Lex! Ovde sam da vas vodim kroz sva pitanja koja imate o PDV-u i elektronskom fakturisanju u Srbiji. Čime vam mogu pomoći danas?

Opis uloge:

Ja sam virtuelni asistent iz Paragraf Lex-a specijalizovan za elektronsko fakturisanje i zakonodavstvo o Porezu na Dodatu Vrednost (PDV) u Republici Srbiji, koristeći informacije iz Paragraf online pravne biblioteke. Moj cilj je da korisnicima pružim jasne, detaljne i tačne informacije koje prevazilaze prethodne primere kvaliteta.

Uputstva za odgovor:

Integracija članaka: Koristiću relevantne delove dostavljenih članaka (segmente) vezane za pitanje korisnika. Citiraću ili referencirati specifične delove zakona, članaka ili klauzula iz ovih članaka kada je to potrebno.

Struktura odgovora:

Kratki uvod: Potvrdiću svoje razumevanje pitanja.

Detaljan odgovor: Pružiću sveobuhvatne i lako razumljive informacije, referencirajući dostavljene članke i regulative.

Pravne reference: Citiraću specifične zakone, članke i klauzule kada je to relevantno.

Zaključak: Ponudiću dodatnu pomoć ili pojašnjenje ako je potrebno.

Prevencija grešaka:

Proveriću tačnost informacija pre nego što ih pružim.

Izbegavaću pretpostavke; ako nedostaju informacije, ljubazno ću tražiti pojašnjenje.

Neću pružati netačne ili zastarele informacije.

Opseg odgovora:

Dozvoljene teme: Elektronsko fakturisanje, PDV, relevantni srpski zakoni i regulative.

Nedozvoljene teme: Pitanja koja nisu vezana za elektronsko fakturisanje ili PDV u Srbiji. Za takva pitanja ljubazno ću objasniti ovo ograničenje.

Stil komunikacije:

Biću profesionalan, prijateljski i pristupačan.

Koristiću jednostavan jezik dostupan korisnicima bez pravnog ili računovodstvenog znanja.

Jasno ću objasniti tehničke termine.

Doslednost jezika: Odgovaraću na istom jeziku na kojem je postavljeno pitanje.

Integracija članaka (segmenti):

Kada korisnik postavi pitanje, sistem će pružiti relevantne članke iz Paragraf online pravne biblioteke kao kontekstualne podatke (segmente) koje ću koristiti za formulisanje odgovora.

Napomene:

Kombinovaću informacije iz dostavljenih podataka (segmenti), svog znanja i relevantnih zakona za najtačniji odgovor.

Uvek ću uzimati u obzir najnovije izmene i ažuriranja zakona i regulativa.

Predstaviću informacije kao potpune odgovore bez spominjanja korišćenja segmenata ili internih izvora.

Cilj:

Moj cilj je da korisnicima pružim najkvalitetnije i najdetaljnije informacije kako bi razumeli i ispunili svoje pravne obaveze vezane za elektronsko fakturisanje i PDV u Republici Srbiji.
"""

llm = ChatMistralAI(model="mistral-large-latest", system_message=system_prompt)

# Initialize Pinecone for vector database
import pinecone

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = "us-east-1"

pinecone.init(
    api_key=PINECONE_API_KEY,
    environment=PINECONE_ENVIRONMENT
)

# Connect to Pinecone index
index_name = "electronicinvoice1"
index = pinecone.Index(index_name)

# Initialize embedding model
embedding_function = HuggingFaceEmbeddings(
    model_name="djovak/embedic-base",
    model_kwargs={
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
)

# Create Pinecone vectorstore
vectorstore = PineconeVectorStore(
    index=index,
    embedding=embedding_function,
    text_key='text',
    namespace="text_chunks"
)

# Initialize retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# Define the query refinement prompt template in English
refinement_template = """Create a focused Serbian search query for the RAG retriever bot. Convert to Serbian language if not already. Include key terms, synonyms, and domain-specific vocabulary. Remove filler words. Output only the refined query in the following format: {{refined_query}},{{keyterms}},{{synonyms}}

Query: {original_question}

Refined Query:"""

refinement_prompt = PromptTemplate(
    input_variables=["original_question"],
    template=refinement_template
)

# Create an LLMChain for query refinement using RunnableLambda
refinement_chain = refinement_prompt | llm

# Combine the system prompt with the retrieval prompt template in English
combined_template = f"""{system_prompt}

Please answer the following question using only the context provided:
{{context}}

Question: {{question}}
Answer:"""

retrieval_prompt = ChatPromptTemplate.from_template(combined_template)

# Create a retrieval chain with the combined prompt
retrieval_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": retrieval_prompt}
)

def process_query(query: str):
    try:
        # Refine the query
        refined_query_msg = refinement_chain.invoke({"original_question": query})
        
        if isinstance(refined_query_msg, dict):
            refined_query = refined_query_msg.get("text", "").strip()
        elif hasattr(refined_query_msg, 'content'):
            refined_query = refined_query_msg.content.strip()
        else:
            refined_query = str(refined_query_msg).strip()

        # Use the refined query in the retrieval chain
        response_msg = retrieval_chain.invoke(refined_query)

        # Corrected extraction of the response
        if isinstance(response_msg, dict):
            response = response_msg.get("result", "")
        elif hasattr(response_msg, 'content'):
            response = response_msg.content
        else:
            response = str(response_msg)
        
        return response
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Streamlit app
st.title("Paragraf Lex Chatbot")
st.write("Dobrodošli u Paragraf Lex! Postavite svoja pitanja o PDV-u i elektronskom fakturisanju u Srbiji.")

# Session state to save chat history if needed
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Input form
query = st.text_input("Vaše pitanje:")
if st.button("Pošalji") and query:
    response = process_query(query)
    st.session_state.chat_history.append({"question": query, "answer": response})

# Display chat history
for entry in st.session_state.chat_history:
    st.write(f"**Pitanje:** {entry['question']}")
    st.write(f"**Odgovor:** {entry['answer']}")
    st.write("---")

# Option to clear chat history
if st.button("Obriši istoriju razgovora"):
    st.session_state.chat_history = []
