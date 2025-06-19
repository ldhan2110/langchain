import "dotenv/config";
import { PGVectorStore } from "@langchain/community/vectorstores/pgvector";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { Ollama, OllamaEmbeddings } from "@langchain/ollama";
import { END, START, Annotation, StateGraph } from "@langchain/langgraph";
import { pull } from "langchain/hub";
// Define embedded models
const embeddings = new OllamaEmbeddings({
    model: "nomic-embed-text", // Default value
    maxRetries: 2,
});
// Define ChatModel
const llm = new Ollama({
    baseUrl: "http://localhost:11434",
    model: "qwen2.5:7b-instruct",
});
// Define vector Store using PGVector
const vectorStore = await PGVectorStore.initialize(embeddings, {
    tableName: "items",
    postgresConnectionOptions: {
        host: "localhost",
        port: 5432,
        user: "myuser",
        password: "mypassword",
        database: "mydatabase",
    },
    columns: {
        idColumnName: "id",
        vectorColumnName: "vector",
        contentColumnName: "content",
        metadataColumnName: "metadata",
    },
    distanceStrategy: "cosine",
});
// // Load the PDF documents
// const loader = new PDFLoader("./Jayden.pdf");
// const docs = await loader.load();
// // Split text into chunks
// const splitter = new RecursiveCharacterTextSplitter({
//     chunkSize: 1000,
//     chunkOverlap: 200,
// });
// const allSplits = await splitter.splitDocuments(docs);
// // Index chunks
// await vectorStore.addDocuments(allSplits);


// Define prompt for question-answering
const promptTemplate = await pull("rlm/rag-prompt");
// Define state for application
const StateAnnotation = Annotation.Root({
    question: (Annotation),
    context: (Annotation),
    answer: (Annotation),
});
const InputStateAnnotation = Annotation.Root({
    question: (Annotation),
});
// Define application steps
const retrieve = async (state) => {
    const retrievedDocs = await vectorStore.similaritySearch(state.question);
    return { context: retrievedDocs };
};
const generate = async (state) => {
    const docsContent = state.context
        .map((doc) => doc.pageContent)
        .join("\n");
    const messages = await promptTemplate.invoke({
        question: state.question,
        context: docsContent,
    });
    const response = await llm.invoke(messages);
    return { answer: response };
};
// Compile application and test
const graph = new StateGraph(StateAnnotation)
    .addNode("retrieve", retrieve)
    .addNode("generate", generate)
    .addEdge(START, "retrieve")
    .addEdge("retrieve", "generate")
    .addEdge("generate", END)
    .compile();

// Invoke Chat
let inputs = { question: "What Mr. Jayden email?" };
const result = await graph.invoke(inputs);
console.log(result.answer);
await vectorStore.end();