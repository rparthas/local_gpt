import chainlit as cl
import os
import logging
from dotenv import load_dotenv
import chromadb
from sentence_transformers import SentenceTransformer
import PyPDF2
from io import BytesIO
from typing import List, Dict, Any
import uuid
import io
import ollama

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize embedding model
logger.info("Loading embedding model...")
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
logger.info("Embedding model loaded successfully")

# Initialize ChromaDB
logger.info("Initializing ChromaDB...")
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(
    name="pdf_documents",
    metadata={"hnsw:space": "cosine"}
)
logger.info("ChromaDB initialized successfully")


def extract_text_from_pdf(pdf_content: bytes) -> str:
    """Extract text from PDF content."""
    logger.info(f"Starting PDF text extraction, content size: {len(pdf_content)} bytes")
    
    text = ""
    try:
        # Create a BytesIO object from the PDF content
        logger.info("Creating BytesIO stream from PDF content")
        pdf_stream = io.BytesIO(pdf_content)
        
        logger.info("Initializing PyPDF2 reader")
        pdf_reader = PyPDF2.PdfReader(pdf_stream)
        
        num_pages = len(pdf_reader.pages)
        logger.info(f"PDF has {num_pages} pages")
        
        for page_num, page in enumerate(pdf_reader.pages):
            try:
                logger.info(f"Extracting text from page {page_num + 1}/{num_pages}")
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
                    logger.info(f"Page {page_num + 1}: extracted {len(page_text)} characters")
                else:
                    logger.warning(f"Page {page_num + 1}: no text extracted")
            except Exception as e:
                logger.error(f"Error extracting text from page {page_num + 1}: {e}")
                continue
                
    except Exception as e:
        logger.error(f"Error reading PDF: {e}")
        raise Exception(f"Failed to extract text from PDF: {e}")
    
    extracted_length = len(text.strip())
    logger.info(f"PDF text extraction completed. Total characters extracted: {extracted_length}")
    return text.strip()


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Split text into overlapping chunks."""
    logger.info(f"Starting text chunking, input length: {len(text)} characters")
    
    if not text or len(text.strip()) == 0:
        logger.warning("No text provided for chunking")
        return []
        
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        
        # Try to break at sentence boundary
        if end < text_length:
            last_period = chunk.rfind('.')
            if last_period > chunk_size * 0.5:  # Only if period is in latter half
                chunk = chunk[:last_period + 1]
                end = start + last_period + 1
        
        chunk = chunk.strip()
        if chunk:  # Only add non-empty chunks
            chunks.append(chunk)
            logger.debug(f"Created chunk {len(chunks)}: {len(chunk)} characters")
        
        start = end - overlap
        
        if start >= text_length:
            break
    
    logger.info(f"Text chunking completed. Created {len(chunks)} chunks")
    return chunks


def add_pdf_to_vectorstore(pdf_content: bytes, filename: str) -> int:
    """Process PDF and add to vector store."""
    logger.info(f"Starting PDF processing for: {filename}")
    
    try:
        # Extract text
        logger.info(f"Step 1: Extracting text from {filename}")
        text = extract_text_from_pdf(pdf_content)
        
        if not text:
            error_msg = f"No text could be extracted from {filename}"
            logger.error(error_msg)
            raise Exception(error_msg)
        
        logger.info(f"Step 2: Text extraction successful for {filename}")
        
        # Chunk text
        logger.info(f"Step 3: Chunking text for {filename}")
        chunks = chunk_text(text)
        
        if not chunks:
            error_msg = f"No valid text chunks could be created from {filename}"
            logger.error(error_msg)
            raise Exception(error_msg)
        
        logger.info(f"Step 4: Created {len(chunks)} chunks for {filename}")
        
        # Generate embeddings
        logger.info(f"Step 5: Generating embeddings for {len(chunks)} chunks from {filename}")
        try:
            embeddings = embedding_model.encode(chunks).tolist()
            logger.info(f"Step 6: Embeddings generated successfully for {filename}")
        except Exception as e:
            logger.error(f"Failed to generate embeddings for {filename}: {e}")
            raise
        
        # Create unique IDs for chunks
        logger.info(f"Step 7: Creating unique IDs for chunks from {filename}")
        ids = [f"{filename}_{i}_{uuid.uuid4().hex[:8]}" for i in range(len(chunks))]
        
        # Prepare metadata
        logger.info(f"Step 8: Preparing metadata for {filename}")
        metadatas = [{"filename": filename, "chunk_id": i} for i in range(len(chunks))]
        
        # Add to collection
        logger.info(f"Step 9: Adding {len(chunks)} chunks to vector store for {filename}")
        try:
            collection.add(
                embeddings=embeddings,
                documents=chunks,
                metadatas=metadatas,
                ids=ids
            )
            logger.info(f"Step 10: Successfully added {len(chunks)} chunks to vector store for {filename}")
        except Exception as e:
            logger.error(f"Failed to add chunks to vector store for {filename}: {e}")
            raise
        
        logger.info(f"PDF processing completed successfully for {filename}")
        return len(chunks)
        
    except Exception as e:
        logger.error(f"Error processing PDF {filename}: {e}")
        raise e


def search_documents(query: str, n_results: int = 5) -> Dict[str, Any]:
    """Search for relevant document chunks."""
    logger.info(f"Searching documents for query: '{query[:50]}...' (showing first 50 chars)")
    
    try:
        logger.info("Generating query embedding")
        query_embedding = embedding_model.encode([query]).tolist()
        
        logger.info(f"Querying vector store for {n_results} results")
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=n_results
        )
        
        num_results = len(results["documents"][0]) if results["documents"] else 0
        logger.info(f"Found {num_results} relevant documents")
        
        return {
            "documents": results["documents"][0] if results["documents"] else [],
            "metadatas": results["metadatas"][0] if results["metadatas"] else [],
            "distances": results["distances"][0] if results["distances"] else []
        }
    except Exception as e:
        logger.error(f"Error searching documents: {e}")
        return {"documents": [], "metadatas": [], "distances": []}


@cl.on_chat_start
async def start_chat():
    logger.info("Starting new chat session")
    
    cl.user_session.set(
        "interaction",
        [
            {
                "role": "system",
                "content": """You are a helpful AI assistant with access to a PDF document database. 

When users ask questions about documents, I will provide you with relevant context from the PDFs. Use this context to answer their questions accurately.

If no relevant context is provided, you can answer general questions normally.

You can also help users with:
1. General conversation and questions
2. Document analysis and Q&A based on uploaded PDFs
3. Image analysis (if images are provided)

Always be helpful and accurate in your responses.""",
            }
        ],
    )

    # Check if we have any documents in the database
    try:
        count = collection.count()
        doc_status = f"\n\n📚 Documents in database: {count} chunks"
        logger.info(f"Database contains {count} document chunks")
    except Exception as e:
        doc_status = "\n\n📚 No documents in database yet"
        logger.warning(f"Could not get document count: {e}")

    msg = cl.Message(content="")

    start_message = f"""Hello! I'm your 100% local AI assistant with PDF Q&A capabilities running on DeepSeek-R1.

🤖 I can help you with:
• General questions and conversation
• PDF document Q&A (upload PDFs to get started)
• Image analysis

📎 To use PDF Q&A: Upload a PDF file and ask questions about its content!{doc_status}

How can I help you today?"""

    for token in start_message:
        await msg.stream_token(token)

    await msg.send()
    logger.info("Chat session started successfully")


@cl.step(type="tool")
async def process_query(input_message: str, image=None, pdf_context: str = None):
    """Process user query with optional PDF context."""
    logger.info(f"Processing query: '{input_message[:100]}...' (showing first 100 chars)")
    
    interaction = cl.user_session.get("interaction")
    
    # Prepare the message content
    if pdf_context:
        logger.info("Adding PDF context to query")
        enhanced_message = f"""Context from relevant documents:
{pdf_context}

User question: {input_message}

Please answer the user's question based on the provided context. If the context doesn't contain relevant information, mention that and provide a general response if possible."""
    else:
        logger.info("Processing query without PDF context")
        enhanced_message = input_message

    if image:
        logger.info("Processing query with image")
        interaction.append({"role": "user", "content": enhanced_message, "images": image})
    else:
        interaction.append({"role": "user", "content": enhanced_message})

    logger.info("Sending query to Ollama model")
    response = ollama.chat(model="deepseek-r1:8b", messages=interaction)

    # Store the original user message (not the enhanced one) in interaction history
    interaction[-1]["content"] = input_message
    interaction.append({"role": "assistant", "content": response.message.content})

    logger.info("Query processed successfully")
    return response


@cl.on_message
async def main(message: cl.Message):
    logger.info(f"Received message with {len(message.elements)} elements")
    
    # Handle PDF uploads
    pdf_files = [file for file in message.elements if file.mime == "application/pdf"]
    images = [file for file in message.elements if "image" in file.mime]
    
    logger.info(f"Found {len(pdf_files)} PDF files and {len(images)} images")
    
    # Debug file elements
    for i, element in enumerate(message.elements):
        logger.info(f"Element {i}: name='{element.name}', mime='{element.mime}', size={element.size if hasattr(element, 'size') else 'unknown'}")
        if hasattr(element, 'content'):
            logger.info(f"Element {i} content type: {type(element.content)}, length: {len(element.content) if element.content else 'None'}")
        if hasattr(element, 'path'):
            logger.info(f"Element {i} path: {element.path}")
    
    if pdf_files:
        logger.info("Starting PDF processing workflow")
        processing_msg = cl.Message(content="📄 Processing PDF(s)...")
        await processing_msg.send()
        
        total_chunks = 0
        processed_files = []
        
        for i, pdf_file in enumerate(pdf_files):
            logger.info(f"Processing PDF {i+1}/{len(pdf_files)}: {pdf_file.name}")
            logger.info(f"PDF file attributes: {dir(pdf_file)}")
            
            try:
                # Try different ways to get content
                content = None
                
                # Method 1: Direct content access
                if hasattr(pdf_file, 'content') and pdf_file.content:
                    content = pdf_file.content
                    logger.info(f"Method 1 - Got content from .content: {len(content)} bytes")
                
                # Method 2: Read from path if content is None
                elif hasattr(pdf_file, 'path') and pdf_file.path:
                    logger.info(f"Method 2 - Trying to read from path: {pdf_file.path}")
                    try:
                        with open(pdf_file.path, 'rb') as f:
                            content = f.read()
                        logger.info(f"Method 2 - Read from path: {len(content)} bytes")
                    except Exception as e:
                        logger.error(f"Method 2 failed: {e}")
                
                # Method 3: Check if it's a file-like object
                elif hasattr(pdf_file, 'read'):
                    logger.info("Method 3 - Trying to read as file-like object")
                    try:
                        content = pdf_file.read()
                        logger.info(f"Method 3 - Read as file object: {len(content)} bytes")
                    except Exception as e:
                        logger.error(f"Method 3 failed: {e}")
                
                # Validate content
                if not content:
                    error_msg = f"No content found in {pdf_file.name} using any method"
                    logger.error(error_msg)
                    logger.error(f"Available attributes: {[attr for attr in dir(pdf_file) if not attr.startswith('_')]}")
                    processed_files.append(f"❌ {pdf_file.name}: No content found")
                    continue
                
                # Ensure content is bytes
                if isinstance(content, str):
                    content = content.encode('utf-8')
                    logger.info(f"Converted string content to bytes: {len(content)} bytes")
                
                logger.info(f"PDF {pdf_file.name} has {len(content)} bytes of content")
                
                # Process PDF directly from content
                chunks = add_pdf_to_vectorstore(content, pdf_file.name)
                total_chunks += chunks
                processed_files.append(f"✅ {pdf_file.name}: {chunks} chunks")
                logger.info(f"Successfully processed {pdf_file.name}: {chunks} chunks added")
                
            except Exception as e:
                error_msg = f"Error processing {pdf_file.name}: {str(e)}"
                logger.error(error_msg)
                processed_files.append(f"❌ {pdf_file.name}: Error - {str(e)}")
        
        # Update processing message
        result_message = f"📄 PDF Processing Complete!\n\n" + "\n".join(processed_files)
        if total_chunks > 0:
            result_message += f"\n\n📚 Total chunks added: {total_chunks}\n\nYou can now ask questions about the uploaded documents!"
        
        processing_msg.content = result_message
        await processing_msg.update()
        
        logger.info(f"PDF processing workflow completed. Total chunks added: {total_chunks}")
        return

    # Handle regular messages (with potential PDF Q&A)
    pdf_context = None
    
    # Debug command to check database status
    if message.content and message.content.strip().lower() == '/debug':
        logger.info("Debug command received - checking database status")
        try:
            doc_count = collection.count()
            await cl.Message(f"📊 **Database Status:**\n- Total document chunks: {doc_count}\n- Collection name: {collection.name}\n- Embedding model: sentence-transformers/all-MiniLM-L6-v2").send()
            
            if doc_count > 0:
                # Show some sample documents
                sample_results = collection.get(limit=3)
                sample_info = "**Sample documents:**\n"
                for i, (doc, metadata) in enumerate(zip(sample_results['documents'], sample_results['metadatas'])):
                    filename = metadata.get('filename', 'unknown')
                    preview = doc[:100] + "..." if len(doc) > 100 else doc
                    sample_info += f"{i+1}. From `{filename}`: {preview}\n"
                await cl.Message(sample_info).send()
            else:
                await cl.Message("❌ No documents found in database. Try uploading a PDF first.").send()
        except Exception as e:
            await cl.Message(f"❌ Error checking database: {e}").send()
        return
    
    # Help command
    if message.content and message.content.strip().lower() in ['/help', 'help']:
        help_text = """🤖 **Local GPT with PDF Q&A**

**Features:**
📄 **PDF Q&A**: Upload PDFs and ask questions about their content
🔍 **Smart Context**: Automatically searches PDF content for relevant information

**How it works:**
1. **Upload PDFs**: Use the attachment button to upload documents
2. **Ask Questions**: Ask anything - the system will search PDFs for relevant context
3. **Get Answers**: Receive responses based on document content or general knowledge

**Commands:**
• `/debug` - Check database status and see what PDFs are loaded
• `/help` - Show this help message

**Example Queries:**
• "What does this document say about X?"
• "Summarize the main points from the uploaded paper"
• "Find information about Y in the documents"

Ready to help! 🚀"""
        
        await cl.Message(content=help_text).send()
        return
    
    # Search for relevant documents if this looks like a question
    if message.content and len(message.content.strip()) > 10:
        logger.info("Searching for relevant documents")
        try:
            search_results = search_documents(message.content)
            if search_results["documents"] and len(search_results["documents"]) > 0:
                # Filter results by relevance (distance threshold)
                relevant_docs = []
                for i, (doc, metadata, distance) in enumerate(zip(
                    search_results["documents"], 
                    search_results["metadatas"], 
                    search_results["distances"]
                )):
                    if distance < 0.8:  # Adjust threshold as needed
                        relevant_docs.append(f"[From {metadata['filename']}]: {doc}")
                        logger.info(f"Found relevant document chunk from {metadata['filename']} (distance: {distance:.3f})")
                
                if relevant_docs:
                    pdf_context = "\n\n".join(relevant_docs[:3])  # Limit to top 3 results
                    logger.info(f"Using {len(relevant_docs)} relevant document chunks for context")
                else:
                    logger.info("No relevant documents found within distance threshold")
            else:
                logger.info("No documents found in search results")
        except Exception as e:
            logger.error(f"Search error: {e}")

    # Log final context decision
    if pdf_context:
        logger.info("🎯 WILL USE PDF CONTEXT for this query")
    else:
        logger.info("🚫 NO PDF CONTEXT - proceeding with general query processing")
        logger.info("Reasons this might happen:")
        logger.info("  • No PDFs uploaded/processed successfully")
        logger.info("  • Query doesn't match document content semantically")
        logger.info("  • Distance threshold too strict (currently 0.8)")
        logger.info("  • Database is empty or search failed")

    # Process the query
    if images:
        logger.info("Processing query with images")
        tool_res = await process_query(message.content, [i.path for i in images], pdf_context)
    else:
        tool_res = await process_query(message.content, pdf_context=pdf_context)

    # Stream the response
    msg = cl.Message(content="")

    for token in tool_res.message.content:
        await msg.stream_token(token)
        
    await msg.send()
    logger.info("Message processing completed")
