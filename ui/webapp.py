import streamlit as st
import argparse
from annotated_text import annotated_text
from haystack.document_store.faiss import FAISSDocumentStore
from haystack.retriever.dense import DensePassageRetriever
from haystack.reader.farm import FARMReader
from haystack.pipeline import DocumentSearchPipeline, ExtractiveQAPipeline


def annotate_answer(answer, context):
    start_idx = context.find(answer)
    end_idx = start_idx + len(answer)
    annotated_text(context[:start_idx], (answer, "ANSWER", "#8ef"), context[end_idx:])


@st.cache(hash_funcs={DocumentSearchPipeline: id, ExtractiveQAPipeline: id,},)
def load_model(faiss_index, sql_url, retriever_pt):
    document_store = FAISSDocumentStore.load(
        faiss_index, sql_url=sql_url, index="document"
    )
    retriever = DensePassageRetriever.load(retriever_pt, document_store=document_store)
    reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=True)
    qa_pipe = ExtractiveQAPipeline(reader, retriever)
    ss_pipe = DocumentSearchPipeline(retriever)
    return qa_pipe, ss_pipe


@st.cache(hash_funcs={FAISSDocumentStore: id},)
def load_doc_store(faiss_index, sql_url):
    document_store = FAISSDocumentStore.load(
        faiss_index, sql_url=sql_url, index="document"
    )
    return document_store


@st.cache(hash_funcs={DensePassageRetriever: id, FAISSDocumentStore: id})
def load_retriever(retriever_pt, document_store):
    retriever = DensePassageRetriever.load(retriever_pt, document_store=document_store)
    return retriever


@st.cache(hash_funcs={FARMReader: id},)
def load_reader():
    reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=True)
    return reader


def retrieve_doc(pipe_type, pipe, question, top_k_reader=5, top_k_retriever=5):
    if pipe_type == "QA":
        response_raw = pipe.run(
            question, top_k_retriever=top_k_retriever, top_k_reader=top_k_reader
        )
        # Format response
        result = []
        answers = response_raw["answers"]
        for i in range(len(answers)):
            answer = answers[i]["answer"]
            if answer:
                context = "..." + answers[i]["context"] + "..."
                meta_name = answers[i]["meta"]["name"]
                relevance = round(answers[i]["probability"] * 100, 2)
                result.append(
                    {
                        "context": context,
                        "answer": answer,
                        "source": meta_name,
                        "relevance": relevance,
                    }
                )
    elif pipe_type == "Semantic Search":
        response_raw = pipe.run(question, top_k_retriever=top_k_retriever)
        result = []
        answers = response_raw["documents"]
        for i in range(len(answers)):
            answer = answers[i]["text"]
            if answer:
                context = "..."
                meta_name = answers[i]["meta"]["name"]
                relevance = round(answers[i]["probability"] * 100, 2)
                result.append(
                    {
                        "context": context,
                        "answer": answer,
                        "source": meta_name,
                        "relevance": relevance,
                    }
                )
    return result


def main():
    parser = argparse.ArgumentParser(
        "Command line tool to crawl links to find DOI and download pdfs via DOI from Scihub."
    )
    parser.add_argument("-f", "--faiss_index", help=".index file")
    parser.add_argument("-s", "--sql_url", help="SQL (sqlite) url")
    parser.add_argument("-r", "--retriever", help="retriever dump .pt")
    parser.add_argument("-g", "--gpu", action="store_true", help="use gpu")
    args = parser.parse_args()
    st.write("# Staubsauger QA")
    st.sidebar.header("Options")
    pipe_type = st.selectbox("Pipeline type:", ("QA", "Semantic Search"))
    top_k_reader = st.sidebar.slider(
        "Max. number of answers", min_value=1, max_value=20, value=3, step=1
    )
    top_k_retriever = st.sidebar.slider(
        "Max. number of documents from retriever",
        min_value=1,
        max_value=20,
        value=3,
        step=1,
    )
    question = st.text_input(
        "Please provide your query:", value="Who is the father of Arya Starck?"
    )
    run_query = st.button("Run")
    document_store = load_doc_store(args.faiss_index, args.sql_url)
    retriever = load_retriever(args.retriever, document_store)
    reader = load_reader()
    qa_pipe = ExtractiveQAPipeline(reader, retriever)
    ss_pipe = DocumentSearchPipeline(retriever)
    if run_query:
        with st.spinner(
            "Performing neural search on documents... 🧠 \n "
            "Do you want to optimize speed or accuracy? \n"
            "Check out the docs: https://haystack.deepset.ai/docs/latest/optimizationmd "
        ):
            if pipe_type is "QA":
                results = retrieve_doc(
                    pipe_type,
                    qa_pipe,
                    question,
                    top_k_reader=top_k_reader,
                    top_k_retriever=top_k_retriever,
                )
            else:
                results = retrieve_doc(
                    pipe_type,
                    ss_pipe,
                    question,
                    top_k_reader=top_k_reader,
                    top_k_retriever=top_k_retriever,
                )
        st.write("## Retrieved answers:")
        for result in results:
            annotate_answer(result["answer"], result["context"])
            "**Relevance:** ", result["relevance"], "**Source:** ", result["source"]


if __name__ == "__main__":
    main()
