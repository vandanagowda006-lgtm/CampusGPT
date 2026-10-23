import sys
import hashlib
from pathlib import Path

import streamlit as st

from mermaid_renderer import render_answer


# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
BACKEND_DIR = BASE_DIR / "backend"
UPLOAD_DIR = BASE_DIR / "uploaded_pdfs"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Allow Python to import backend modules.
sys.path.insert(0, str(BACKEND_DIR))

from rag_pipeline import build_rag_pipeline, ask_question


# --------------------------------------------------
# 2. Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="CampusGPT | AI Study Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

CSS_PATH = BASE_DIR / "style.css"

if CSS_PATH.exists():
    with open(CSS_PATH, encoding="utf-8") as css_file:
        st.markdown(
            f"<style>{css_file.read()}</style>",
            unsafe_allow_html=True
        )


# --------------------------------------------------
# 3. Custom styling
# --------------------------------------------------

st.markdown(
    """
    <style>
    .stApp {
        background-color: #f8fafc;
        color: #1e293b;
    }

    h1 {
        color: #1e3a8a !important;
        font-weight: 800;
    }

    h2, h3, h4 {
        color: #1e3a8a;
    }

    .stApp p {
        color: #334155;
    }

    [data-testid="stSidebar"] {
        background-color: #eaf2ff;
    }

    [data-testid="stSidebar"] * {
        color: #1e293b;
    }

    [data-testid="stChatMessage"] {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 16px;
        margin-bottom: 12px;
    }

    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
        border: 1px solid #bfdbfe;
    }

    .stButton > button:hover {
        border-color: #2563eb;
        color: #1d4ed8;
    }

    [data-testid="stFileUploader"] {
        border-radius: 12px;
    }

    .feature-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 20px;
        min-height: 150px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
    }

    .feature-card h3 {
        color: #1d4ed8;
        margin-bottom: 10px;
    }

    .feature-card p {
        color: #475569;
        font-size: 14px;
    }

    .welcome-text {
        text-align: center;
        color: #64748b;
        font-size: 18px;
        margin-bottom: 25px;
    }

    .footer {
        text-align: center;
        color: #64748b;
        font-size: 13px;
        padding: 20px 0;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# --------------------------------------------------
# 4. Sidebar and multiple PDF upload
# --------------------------------------------------

with st.sidebar:

    st.markdown(
        """
        <div style="
            font-family: 'Space Grotesk', sans-serif;
            font-size: 23px;
            font-weight: 700;
            color: #ffffff;
            letter-spacing: -0.5px;
            white-space: nowrap;
            margin-bottom: 15px;
        ">
            📚 Your Documents
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("Upload your study materials to get started.")

    uploaded_pdfs = st.file_uploader(
        "Choose PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload your notes, textbooks, or study materials."
    )

    if uploaded_pdfs:
        st.success(
            f"Selected {len(uploaded_pdfs)} PDF file(s)."
        )

        for pdf in uploaded_pdfs:
            st.write(f"📄 {pdf.name}")

    else:
        st.info("No PDFs uploaded yet.")

    st.divider()

    st.markdown(
        """
        <div style="
            font-family: 'Space Grotesk', sans-serif;
            font-size: 18px;
            font-weight: 700;
            color: #c084fc;
            margin: 12px 0;
        ">
            ✨ What you can do
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        - 📄 Upload multiple study PDFs
        - 🔍 Ask questions across documents
        - 💡 Understand difficult topics
        - 📖 Find information faster
        - 📊 View supported Mermaid diagrams
        """
    )

    st.divider()

    st.caption("CampusGPT | AI Study Assistant")


# --------------------------------------------------
# 5. Main heading
# --------------------------------------------------

st.title("🎓 CampusGPT")

st.markdown(
    """
    <div class="welcome-text">
        Your personal AI study assistant.<br>
        Upload your study materials, ask questions,
        and learn smarter.
    </div>
    """,
    unsafe_allow_html=True
)


# --------------------------------------------------
# 6. Welcome screen when no PDFs are uploaded
# --------------------------------------------------

if not uploaded_pdfs:

    st.markdown(
        """
        <div class="hero-panel">
            <div class="eyebrow">💜 YOUR PERSONAL STUDY SPACE</div>

            <h2>
                Your next academic comeback
                <span style="color:#c084fc;">starts here.</span>
            </h2>

            <p>
                Upload your notes. Ask anything.
                Understand the hard stuff without the headache.
            </p>

            <p>
                ✦ PDF-powered answers &nbsp;
                ✦ Smarter revision &nbsp;
                ✦ Less study stress
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### ✨ Your study era starts now")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
            <div class="feature-card">
                <h3>📚 Notes in, magic out</h3>
                <p>
                    Upload your PDFs and make your
                    study material easier to understand.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            """
            <div class="feature-card">
                <h3>🧠 Ask away</h3>
                <p>
                    Ask questions and get answers
                    based on your documents.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            """
            <div class="feature-card">
                <h3>⚡ Study smarter</h3>
                <p>
                    Find useful information faster
                    and spend more time learning.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown(
        """
        <div class="motivation-banner">
            ✨ Main character energy. Academic comeback loading...
            <br><br>
            Your goals are big. Let's make studying feel easier. 💜
        </div>

        <div class="footer">
            Made for students, powered by AI. 🎓
        </div>
        """,
        unsafe_allow_html=True
    )

    st.stop()


# --------------------------------------------------
# 7. Save uploaded PDFs and preserve original names
# --------------------------------------------------

saved_pdf_paths = []
document_details = []

for uploaded_pdf in uploaded_pdfs:

    pdf_bytes = uploaded_pdf.getvalue()

    file_hash = hashlib.sha256(pdf_bytes).hexdigest()

    # Hash filenames prevent uploaded files from overwriting each other.
    pdf_path = UPLOAD_DIR / f"{file_hash}.pdf"

    if not pdf_path.exists():
        pdf_path.write_bytes(pdf_bytes)

    saved_pdf_paths.append(str(pdf_path))

    # Preserve the original filename for source citations.
    document_details.append(
        {
            "name": uploaded_pdf.name,
            "hash": file_hash,
            "path": str(pdf_path)
        }
    )


# Create an ID representing the selected PDFs and display names.
combined_hash = hashlib.sha256(
    "|".join(
        f"{item['hash']}:{item['name']}"
        for item in sorted(
            document_details,
            key=lambda item: (item["hash"], item["name"])
        )
    ).encode("utf-8")
).hexdigest()


# --------------------------------------------------
# 8. Build and cache the RAG pipeline
# --------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_campusgpt(
    pdf_paths,
    original_names,
    document_hash,
    pipeline_version="generic-multipdf-mermaid-fix-4"
):
    """
    Build the RAG pipeline and preserve original source filenames.
    """

    return build_rag_pipeline(
        pdf_paths,
        original_names=original_names
    )


# --------------------------------------------------
# 9. Reset chat when selected documents change
# --------------------------------------------------

if st.session_state.get("current_document_hash") != combined_hash:

    st.session_state.current_document_hash = combined_hash
    st.session_state.messages = []


if "messages" not in st.session_state:
    st.session_state.messages = []


# --------------------------------------------------
# 10. Process all selected PDFs
# --------------------------------------------------

with st.spinner(
    "📚 Reading your PDFs and preparing CampusGPT..."
):

    try:

        original_names = tuple(
            item["name"] for item in document_details
        )

        index, chunks = load_campusgpt(
            tuple(saved_pdf_paths),
            original_names,
            combined_hash,
            "generic-multipdf-mermaid-fix-4"
        )

    except Exception as error:

        st.error(f"Could not process your PDFs: {error}")

        st.stop()


# --------------------------------------------------
# 11. Document status
# --------------------------------------------------

st.success(
    f"✅ CampusGPT is ready with {len(uploaded_pdfs)} PDF file(s)!"
)

with st.expander("📚 View uploaded documents"):

    for pdf in uploaded_pdfs:
        st.write(f"📄 {pdf.name}")

st.divider()


# --------------------------------------------------
# 12. Chat controls
# --------------------------------------------------

col1, col2 = st.columns([4, 1])

with col1:
    st.subheader("💬 Chat with your PDFs")

with col2:

    if st.button("🗑️ Clear Chat", use_container_width=True):

        st.session_state.messages = []
        st.rerun()


# --------------------------------------------------
# 13. Display conversation history
# --------------------------------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        if message["role"] == "assistant":

            # Render Markdown and Mermaid diagrams in assistant answers.
            render_answer(message["content"])

        else:

            st.markdown(message["content"])


# --------------------------------------------------
# 14. Ask questions
# --------------------------------------------------

question = st.chat_input(
    "Ask a question about your uploaded documents..."
)

if question:

    # Save and display the user's message.
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    # Generate and display the assistant's answer.
    with st.chat_message("assistant"):

        with st.spinner(
            "🔍 Searching your documents and generating an answer..."
        ):

            try:

                answer = ask_question(
                    index,
                    chunks,
                    question,
                    history=st.session_state.messages[:-1]
                )

                # Render Markdown and Mermaid diagrams inline.
                render_answer(answer)

                # Store the original answer for future conversation turns.
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer
                    }
                )

            except Exception as error:

                st.error(f"Something went wrong: {error}")


# --------------------------------------------------
# 15. Footer
# --------------------------------------------------

st.markdown(
    """
    <div class="footer">
        🎓 CampusGPT · Your personal AI study assistant
    </div>
    """,
    unsafe_allow_html=True
)