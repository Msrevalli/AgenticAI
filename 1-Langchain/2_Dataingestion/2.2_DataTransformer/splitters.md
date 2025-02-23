# **📌 Text Splitters in LangChain**
Text splitters in **LangChain** are used to **break large documents into smaller chunks** for **efficient retrieval, storage, and processing** in **AI applications, RAG pipelines, and vector databases**.

---

## **🚀 Why Use Text Splitters?**
✔ **Improves Retrieval** – Smaller chunks **enhance search accuracy** in AI-powered applications.  
✔ **Optimizes LLM Processing** – Helps **stay within token limits** for LLMs.  
✔ **Reduces Context Window Issues** – Ensures AI gets **relevant chunks** instead of entire documents.  
✔ **Enhances RAG Performance** – Enables **semantic search & question-answering**.  

---

## **🛠 Types of Text Splitters in LangChain**
| **Splitter** | **Best For** | **How It Splits** |
|-------------|-------------|------------------|
| **RecursiveCharacterTextSplitter** | General-purpose AI applications | Splits at the largest possible **natural boundary** (paragraph → sentence → word) |
| **CharacterTextSplitter** | Basic text splitting | Splits at a specific **character** (e.g., `\n`, `.`, ` `) |
| **TokenTextSplitter** | LLMs like GPT, Claude | Splits text based on **token count** |
| **NLTKTextSplitter** | NLP-based AI models | Uses **NLTK sentence tokenization** |
| **MarkdownTextSplitter** | Processing Markdown files | Splits **headers, paragraphs, lists, and code blocks** |
| **HTMLHeaderTextSplitter** | Extracting web content | Splits HTML using **headers (H1, H2, etc.)** |
| **LanguageTextSplitter** | Code analysis | Splits **Python, Java, C++ code** into logical sections |
| **SentenceTransformersTextSplitter** | Semantic chunking | Uses **pre-trained models for intelligent splitting** |

---

## **1️⃣ Recursive Character Splitter (Recommended)**
🔹 **Best for:** **General-purpose document chunking**  
🔹 **How it works:** **Splits at natural boundaries** (paragraph → sentence → word).  

### **📌 Example**
```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

text = "LangChain makes AI applications powerful. It supports retrieval, agents, and chains."

splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=10)
chunks = splitter.split_text(text)

print(chunks)
```
✔ **Best for:** AI-powered **search, summarization, and chatbots**.

---

## **2️⃣ Character Text Splitter**
🔹 **Best for:** **Splitting at a specific character** (`\n`, `.`, `,`)  
🔹 **How it works:** Splits **whenever a character limit is reached**.  

### **📌 Example**
```python
from langchain.text_splitter import CharacterTextSplitter

splitter = CharacterTextSplitter(separator="\n\n", chunk_size=100, chunk_overlap=10)
chunks = splitter.split_text("Paragraph 1...\n\nParagraph 2...\n\nParagraph 3...")
print(chunks)
```
✔ **Best for:** **Simple text processing**.

---

## **3️⃣ Token Text Splitter**
🔹 **Best for:** **LLMs like GPT-4, Claude, and Mistral**  
🔹 **How it works:** Splits **based on token count** instead of characters.  

### **📌 Example**
```python
from langchain.text_splitter import TokenTextSplitter

splitter = TokenTextSplitter(chunk_size=100, chunk_overlap=10)
chunks = splitter.split_text("OpenAI's GPT models use tokens, not words.")
print(chunks)
```
✔ **Best for:** **GPT & LLM-based applications**.

---

## **4️⃣ NLTK Sentence Splitter**
🔹 **Best for:** **Splitting text at sentence boundaries**  
🔹 **How it works:** Uses **NLTK’s sentence tokenizer**.  

### **📌 Example**
```python
from langchain.text_splitter import NLTKTextSplitter

splitter = NLTKTextSplitter()
chunks = splitter.split_text("Sentence one. Sentence two! Sentence three?")
print(chunks)
```
✔ **Best for:** **Natural Language Processing (NLP) tasks**.

---

## **5️⃣ Markdown Text Splitter**
🔹 **Best for:** **Processing Markdown files**  
🔹 **How it works:** Splits **headers, code blocks, lists, and paragraphs**.  

### **📌 Example**
```python
from langchain.text_splitter import MarkdownTextSplitter

text = "# Header 1\nContent under header.\n## Header 2\nMore content."
splitter = MarkdownTextSplitter()
chunks = splitter.split_text(text)
print(chunks)
```
✔ **Best for:** **AI-powered document processing**.

---

## **6️⃣ HTML Header Splitter**
🔹 **Best for:** **Extracting structured content from web pages**  
🔹 **How it works:** Splits **at HTML headers (H1, H2, H3, etc.)**.  

### **📌 Example**
```python
from langchain.text_splitter import HTMLHeaderTextSplitter

html_text = "<h1>Main Title</h1><p>Intro paragraph.</p><h2>Subsection</h2><p>More content.</p>"
splitter = HTMLHeaderTextSplitter(headers_to_split_on=["h1", "h2"])
chunks = splitter.split_text(html_text)
print(chunks)
```
✔ **Best for:** **Processing scraped web content**.

---

## **7️⃣ Code (Language) Splitter**
🔹 **Best for:** **Splitting programming code into functions, classes, and logic blocks**.  
🔹 **How it works:** Splits **Python, Java, C++ code** using language rules.  

### **📌 Example**
```python
from langchain.text_splitter import Language, LanguageTextSplitter

code = "def function():\n    print('Hello World')\n\nclass MyClass:\n    pass"
splitter = LanguageTextSplitter(language=Language.PYTHON, chunk_size=50, chunk_overlap=10)
chunks = splitter.split_text(code)
print(chunks)
```
✔ **Best for:** **AI-powered code analysis & documentation**.

---

## **8️⃣ Sentence Transformers Splitter**
🔹 **Best for:** **Semantic chunking based on sentence meaning**  
🔹 **How it works:** Uses **pre-trained sentence transformers** to split intelligently.  

### **📌 Example**
```python
from langchain.text_splitter import SentenceTransformersTextSplitter

splitter = SentenceTransformersTextSplitter(model_name="sentence-transformers/all-MiniLM-L6-v2")
chunks = splitter.split_text("Machine Learning is transforming industries. AI is growing fast.")
print(chunks)
```
✔ **Best for:** **Semantic AI applications**.

---

## **🚀 Which LangChain Splitter Should You Use?**
| **Splitter** | **Best For** |
|-------------|-------------|
| **RecursiveCharacterTextSplitter** | **General-purpose document splitting** |
| **CharacterTextSplitter** | **Simple text-based chunking** |
| **TokenTextSplitter** | **GPT-4, Claude, LLMs (token-based chunking)** |
| **NLTKTextSplitter** | **NLP-based sentence tokenization** |
| **MarkdownTextSplitter** | **Processing Markdown files** |
| **HTMLHeaderTextSplitter** | **Extracting structured content from web pages** |
| **LanguageTextSplitter** | **Splitting Python, Java, and C++ code** |
| **SentenceTransformersTextSplitter** | **Semantic-aware chunking** |

---

