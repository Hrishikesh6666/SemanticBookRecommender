import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv

# Load environment variables from .env (if needed for private Hugging Face models)
load_dotenv()
hf_api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")  # Optional for public models

# Import document loader and text splitter
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter

# Import the Chroma vector store
from langchain_community.vectorstores import Chroma

# Import HuggingFaceEmbeddings from LangChain's Hugging Face integration
from langchain_huggingface import HuggingFaceEmbeddings

import gradio as gr

# -------------------------------
# 1. Data and Document Processing
# -------------------------------

# Load the books dataset (ensure "books_with_emotions.csv" is in your working directory)
books = pd.read_csv("books_with_emotions.csv")

# Create a column for large thumbnails (append parameter or use placeholder if missing)
books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
books["large_thumbnail"] = np.where(
    books["large_thumbnail"].isna(),
    "cover-not-found.jpg",
    books["large_thumbnail"],
)

# Load the text documents (ensure "tagged_description.txt" exists and is UTF-8 encoded)
raw_documents = TextLoader("tagged_description.txt", encoding="utf-8").load()

# Split the documents into smaller chunks
text_splitter = CharacterTextSplitter(separator="\n", chunk_size=0, chunk_overlap=0)
documents = text_splitter.split_documents(raw_documents)

# ------------------------------------
# 2. Initialize Hugging Face Embeddings
# ------------------------------------

# Use a public SentenceTransformer model that runs locally.
# Here we use "sentence-transformers/all-mpnet-base-v2" which does not require an API key.
model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {'device': 'cpu'}  # Change to 'cuda' if using a GPU
encode_kwargs = {'normalize_embeddings': False}

# Create the embeddings instance
hf_embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

# Build the vector store from the document chunks
db_books = Chroma.from_documents(documents, hf_embeddings)


# ------------------------------------
# 3. Recommendation and Query Functions
# ------------------------------------

def retrieve_semantic_recommendations(
        query: str,
        category: str = None,
        tone: str = None,
        initial_top_k: int = 50,
        final_top_k: int = 16,
) -> pd.DataFrame:
    # Perform similarity search using the vector store
    recs = db_books.similarity_search(query, k=initial_top_k)

    # Extract book IDs from search results (assumes rec.page_content begins with a numeric identifier)
    books_list = [int(rec.page_content.strip('"').split()[0]) for rec in recs]

    # Filter the books dataset by matching ISBN13 values
    book_recs = books[books["isbn13"].isin(books_list)].head(initial_top_k)

    # Apply category filtering if provided
    if category and category != "All":
        book_recs = book_recs[book_recs["simple_categories"] == category].head(final_top_k)
    else:
        book_recs = book_recs.head(final_top_k)

    # Optionally sort by tone (mapping tone to column name in the CSV)
    tone_mapping = {
        "Happy": "joy",
        "Surprising": "surprise",
        "Angry": "anger",
        "Suspenseful": "fear",
        "Sad": "sadness"
    }
    if tone in tone_mapping:
        book_recs.sort_values(by=tone_mapping[tone], ascending=False, inplace=True)

    return book_recs


def recommend_books(query: str, category: str, tone: str):
    recommendations = retrieve_semantic_recommendations(query, category, tone)
    results = []
    for _, row in recommendations.iterrows():
        # Truncate the book description to the first 30 words
        description = row["description"]
        truncated_desc_split = description.split()
        truncated_description = " ".join(truncated_desc_split[:30]) + "..."

        # Format author names
        authors_split = row["authors"].split(";")
        if len(authors_split) == 2:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
        else:
            authors_str = row["authors"]

        # Create caption for display
        caption = f"{row['title']} by {authors_str}: {truncated_description}"
        results.append((row["large_thumbnail"], caption))

    return results


# ------------------------------------
# 4. Gradio Dashboard Setup
# ------------------------------------

# Prepare dropdown choices for categories and emotional tones
categories = ["All"] + sorted(books["simple_categories"].unique())
tones = ["All"] + ["Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

with gr.Blocks(theme=gr.themes.Glass()) as dashboard:
    gr.Markdown("# 📚 Semantic Book Recommender")

    with gr.Row():
        user_query = gr.Textbox(
            label="Enter a book description:",
            placeholder="e.g., A story about forgiveness"
        )
        category_dropdown = gr.Dropdown(
            choices=categories,
            label="Select a category:",
            value="All"
        )
        tone_dropdown = gr.Dropdown(
            choices=tones,
            label="Select an emotional tone:",
            value="All"
        )
        submit_button = gr.Button("🔍 Find Recommendations")

    gr.Markdown("## 🎯 Recommended Books")
    output = gr.Gallery(label="Recommended Books", columns=8, rows=2)

    submit_button.click(
        fn=recommend_books,
        inputs=[user_query, category_dropdown, tone_dropdown],
        outputs=output
    )

# ------------------------------------
# 5. Launch the Dashboard
# ------------------------------------

if __name__ == "__main__":
    dashboard.launch()
