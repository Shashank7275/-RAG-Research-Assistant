import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Enterprise RAG Assistant",
    page_icon="📚",
    layout="wide",
)

st.title("Enterprise RAG Research Assistant")
st.caption(
    "Research across your documents with hybrid search "
    "and source-grounded answers."
)

with st.sidebar:
    st.header("Document Management")

    uploaded_file = st.file_uploader(
        "Upload a research PDF (up to 200 MB)",
        type=["pdf"],
    )

    if uploaded_file and st.button(
        "Upload and Index",
        type="primary",
        use_container_width=True,
    ):
        with st.spinner("Processing document..."):
            try:
                response = requests.post(
                    f"{API_URL}/documents",
                    files={
                        "file": (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            "application/pdf",
                        )
                    },
                    timeout=600,
                )

                if response.ok:
                    result = response.json()

                    if result["status"] == "indexed":
                        st.success(
                            f"Indexed {result['chunks']} chunks."
                        )
                    else:
                        st.info("This PDF is already indexed.")

                else:
                    try:
                        err_text = response.json().get("detail", response.text)
                    except Exception:
                        err_text = response.text
                    st.error(f"Upload failed: {err_text}")

            except requests.RequestException as exc:
                st.error(f"Backend connection failed: {exc}")

    if st.button("Refresh document list"):
        st.session_state["refresh_docs"] = True

    try:
        response = requests.get(
            f"{API_URL}/documents",
            timeout=15,
        )
        response.raise_for_status()
        documents = response.json()

        st.subheader("Indexed Documents")

        if documents:
            for doc in documents:
                st.markdown(f"**{doc['filename']}**")
                st.caption(
                    f"{doc['pages']} pages · "
                    f"{doc['chunks']} chunks"
                )
        else:
            st.info("Upload your first research PDF.")

    except requests.RequestException:
        st.warning("Start the FastAPI backend first.")

st.divider()
st.subheader("Ask your research question")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message.get("sources"):
            with st.expander("View source passages"):
                for source in message["sources"]:
                    st.markdown(
                        f"**[{source['citation']}] "
                        f"{source['source']} — "
                        f"Page {source['page']}**"
                    )
                    st.write(source["excerpt"])

question = st.chat_input(
    "Ask something about your documents..."
)

if question:
    st.session_state.messages.append({
        "role": "user",
        "content": question,
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching and generating answer..."):
            try:
                response = requests.post(
                    f"{API_URL}/ask",
                    json={"question": question},
                    timeout=180,
                )

                if not response.ok:
                    try:
                        err_msg = response.json().get("detail", response.text)
                    except Exception:
                        err_msg = response.text
                    st.error(f"Error ({response.status_code}): {err_msg}")
                else:
                    result = response.json()
                    answer = result["answer"]
                    sources = result.get("sources", [])

                    st.markdown(answer)

                    if sources:
                        with st.expander("View source passages"):
                            for source in sources:
                                st.markdown(
                                    f"**[{source['citation']}] "
                                    f"{source['source']} — "
                                    f"Page {source['page']}**"
                                )
                                st.write(source["excerpt"])

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                    })

            except requests.RequestException as exc:
                st.error(f"Request failed: {exc}")
