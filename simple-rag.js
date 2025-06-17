import 'dotenv/config';
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { Ollama, OllamaEmbeddings } from "@langchain/ollama";
import { createRetrievalChain } from "langchain/chains/retrieval";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { ChatPromptTemplate } from "@langchain/core/prompts";

// Define embedded models
const embeddings  = new OllamaEmbeddings({
  model: "nomic-embed-text", // Default value
  temperature: 0,
  maxRetries: 2,
});

const llm = new Ollama({
  baseUrl: "http://localhost:11434",
   model: "qwen2.5:7b-instruct",
});

// Load the PDF documents
const loader = new PDFLoader("./Jayden.pdf");
const docs = await loader.load();

// Splits the document based on chunks size & chunk overlap
const textSplitter = new RecursiveCharacterTextSplitter({
  chunkSize: 500,
  chunkOverlap: 200,
});
const allSplits = await textSplitter.splitDocuments(docs);


// Define Vector Store
const vectorStore = new MemoryVectorStore(embeddings);
await vectorStore.addDocuments(allSplits);

// Define Questions Store
async function performSimilaritySearch(vectorStore, query, k = 3) {
  try {
    // Perform similarity search
    const results = await vectorStore.similaritySearch(query, k);
    console.log(`\nSimilarity search results for: "${query}"`);
    results.forEach((doc, index) => {
      console.log(`\nResult ${index + 1}:`);
      console.log(`Content: ${doc.pageContent}`);
      console.log(`Metadata: ${JSON.stringify(doc.metadata)}`);
    });
    return results;
  } catch (error) {
    console.error("Error performing similarity search:", error);
    throw error;
  }
}

async function createQAChain(vectorStore) {
  try {
    // Create a prompt template
    const prompt = ChatPromptTemplate.fromTemplate(`
      Answer the following question based only on the provided context:
      Context: {context}
      Question: {input}
      Answer:`);
    // Create document chain
    const documentChain = await createStuffDocumentsChain({
      llm,
      prompt,
    });
    // Create retrieval chain
    const retrievalChain = await createRetrievalChain({
      combineDocsChain: documentChain,
      retriever: vectorStore.asRetriever(),
    });
    return retrievalChain;
  } catch (error) {
    console.error("Error creating QA chain:", error);
    throw error;
  }
}

async function askQuestion(chain, question) {
  try {
    const response = await chain.invoke({
      input: question
    });
    console.log(`\nQuestion: ${question}`);
    console.log(`Answer: ${response.answer}`);
    console.log(`Context used: ${response.context.map(doc => doc.pageContent).join(', ')}`);
    return response;
  } catch (error) {
    console.error("Error asking question:", error);
    throw error;
  }
}

// Ask Question
// await performSimilaritySearch(vectorStore, "What is Mr.Jayden email ?");

// // Create QA chain
const qaChain = await createQAChain(vectorStore);

// // Ask questions
// await askQuestion(qaChain, "Did Mr.Jayden work at Vuong Khuong?");
await askQuestion(qaChain, "What is Mr.Jayden full name ?");