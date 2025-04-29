# Hoax News Fact Checking

A tool for detecting hoax/fake news in Indonesia using RAG (Retrieval-Augmented Generation).

## Features

- CLI interface for fact checking
- Web interface using Gradio
- API endpoints via FastAPI
- Vector database (Milvus) for storing document embeddings
- Web search integration (Tavily)

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up Milvus:
   - Ensure you have Milvus installed and running. Refer to the Milvus documentation for installation instructions.
   - The application connects to Milvus using default settings. If your Milvus instance uses different settings, adjust the connection parameters in `hoax_detect/services/vector_store.py`.

## Usage

### CLI

1.  **Initialize the vector database (Optional):**
    If you want to use the vector database for fact-checking, you need to initialize it first. This involves loading the dataset and creating embeddings. This step is only required if you haven't already initialized the database or if you want to refresh the data.

    ```bash
    python -m hoax_detect.data.loader --init_db
    ```

2.  **Run the command line interface:**

    ```bash
    python -m hoax_detect.cli --query "your query here" --use_vector_db True --use_tavily True
    ```

    -   `--query`: The news snippet or statement you want to fact-check.  **Required**.
    -   `--use_vector_db`:  A boolean value indicating whether to use the vector database for retrieving context. Defaults to `True`.
    -   `--use_tavily`: A boolean value indicating whether to use Tavily for web search. Defaults to `True`.

    Example:

    ```bash
    python -m hoax_detect.cli --query "Jokowi resigns" --use_vector_db True --use_tavily True
    ```

### Gradio App

1.  **Run the Gradio application:**

    ```bash
    python gradio_app.py
    ```

2.  **Access the application in your browser:**

    The application will provide a local URL (usually `http://localhost:7860`) that you can use to access the Gradio interface in your web browser.

3.  **Use the interface:**

    -   Enter the news snippet or statement you want to fact-check in the input field.
    -   Select whether to use the vector database and/or Tavily for context retrieval using the provided checkboxes.
    -   Click the "Submit" button to initiate the fact-checking process.
    -   The results, including the fact-checked statement and supporting evidence, will be displayed in the output area.

## API

The application also provides a FastAPI API.  Refer to `hoax_detect/api.py` for details on available endpoints.  You can access the API documentation at `/docs` after running the API.

## Configuration

Configuration settings, such as the dataset path, are defined in `hoax_detect/config.py`. You can modify these settings by creating a `.env` file in the project root directory. See `.env.example` for the available options.
