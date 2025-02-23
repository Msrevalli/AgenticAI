### **📌 Data Loaders in LangChain**
Data loaders in **LangChain** are used to **extract and load data** from different sources, such as **documents, databases, APIs, cloud storage, and web pages**. They play a crucial role in **Retrieval-Augmented Generation (RAG), AI search, and knowledge retrieval**.

---

## **🚀 Categories of LangChain Data Loaders**
| **Category** | **Examples** | **Usage** |
|-------------|-------------|----------|
| **File Loaders** | PDF, Word, CSV, Excel, JSON, Text | Extract text from documents |
| **Database Loaders** | SQL, NoSQL, Firebase, Google Sheets | Fetch structured data |
| **Web & API Loaders** | Web scraping, RSS, REST APIs | Extract web data |
| **Cloud Storage Loaders** | AWS S3, Google Drive, Azure Blob | Fetch data from cloud |
| **Multimedia Loaders** | Images, Audio, Video | Extract text using OCR |

---

## **🛠 1. File-Based Data Loaders**
Extracts text from **PDFs, Word, CSV, JSON, Excel, and text files**.

### **📌 PDF Loader**
```python
from langchain.document_loaders import PyPDFLoader

loader = PyPDFLoader("sample.pdf")
documents = loader.load()
print(documents[0].page_content)
```
✔ **Best for:** Extracting text from **PDF files**.

---

### **📌 Word (DOCX) Loader**
```python
from langchain.document_loaders import Docx2txtLoader

loader = Docx2txtLoader("document.docx")
documents = loader.load()
print(documents[0].page_content)
```
✔ **Best for:** Extracting text from **Microsoft Word documents**.

---

### **📌 CSV Loader**
```python
from langchain.document_loaders import CSVLoader

loader = CSVLoader("data.csv")
documents = loader.load()
print(documents[0].page_content)
```
✔ **Best for:** Extracting structured data from **spreadsheets**.

---

### **📌 JSON Loader**
```python
from langchain.document_loaders import JSONLoader

loader = JSONLoader(file_path="data.json", jq_schema=".data[]")
documents = loader.load()
print(documents[0].page_content)
```
✔ **Best for:** Extracting text from **JSON-based APIs & datasets**.

---

## **🛠 2. Database Loaders**
Extracts data from **SQL, NoSQL, Firebase, Google Sheets**.

### **📌 SQL Database Loader**
```python
from langchain.document_loaders import SQLDatabaseLoader
from langchain.sql_database import SQLDatabase

db = SQLDatabase.from_uri("sqlite:///sample.db")
loader = SQLDatabaseLoader(db, query="SELECT * FROM users")
documents = loader.load()
print(documents[0].page_content)
```
✔ **Best for:** Fetching structured data from **databases**.

---

### **📌 Google Sheets Loader**
```python
from langchain.document_loaders import GoogleSheetsLoader

loader = GoogleSheetsLoader(spreadsheet_id="your_sheet_id", credentials_path="credentials.json")
documents = loader.load()
print(documents[0].page_content)
```
✔ **Best for:** Extracting structured data from **Google Sheets**.

---

## **🛠 3. Web & API Loaders**
Fetches data from **web pages, APIs, RSS feeds, and news sources**.

### **📌 Web Scraping (URL Loader)**
```python
from langchain.document_loaders import UnstructuredURLLoader

urls = ["https://example.com"]
loader = UnstructuredURLLoader(urls=urls)
documents = loader.load()
print(documents[0].page_content)
```
✔ **Best for:** Extracting **text from web pages**.

---

### **📌 API Data Loader**
```python
from langchain.document_loaders import JSONLoader

loader = JSONLoader(file_path="api_response.json", jq_schema=".data[]")
documents = loader.load()
print(documents[0].page_content)
```
✔ **Best for:** Fetching **structured API responses**.

---

## **🛠 4. Cloud Storage Loaders**
Extracts data from **AWS S3, Google Drive, and Azure Blob Storage**.

### **📌 AWS S3 Loader**
```python
from langchain.document_loaders import S3FileLoader

loader = S3FileLoader(bucket="my-bucket", key="file.txt")
documents = loader.load()
print(documents[0].page_content)
```
✔ **Best for:** Fetching data from **Amazon S3**.

---

### **📌 Google Drive Loader**
```python
from langchain.document_loaders import GoogleDriveLoader

loader = GoogleDriveLoader(folder_id="your_folder_id", credentials_path="credentials.json")
documents = loader.load()
print(documents[0].page_content)
```
✔ **Best for:** Extracting files from **Google Drive**.

---

## **🛠 5. Multimedia Loaders**
Extracts text from **images, audio, and videos**.

### **📌 OCR (Image Text Extraction)**
```python
from langchain.document_loaders import UnstructuredImageLoader

loader = UnstructuredImageLoader("image.png")
documents = loader.load()
print(documents[0].page_content)
```
✔ **Best for:** Extracting text from **scanned documents and images**.

---

## **🚀 Which Data Loader Should You Use?**
| **Data Source** | **Best Loader** |
|---------------|---------------|
| **PDFs** | `PyPDFLoader` |
| **Word Docs** | `Docx2txtLoader` |
| **CSV Files** | `CSVLoader` |
| **Databases (SQL, NoSQL)** | `SQLDatabaseLoader` |
| **Web Pages** | `UnstructuredURLLoader` |
| **APIs (JSON, REST)** | `JSONLoader` |
| **Cloud Storage (S3, Drive)** | `S3FileLoader`, `GoogleDriveLoader` |
| **OCR for Images** | `UnstructuredImageLoader` |

---

