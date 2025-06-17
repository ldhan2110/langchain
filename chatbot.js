import { ChatOllama } from "@langchain/ollama";
import { ChatPromptTemplate, MessagesPlaceholder } from "@langchain/core/prompts";
import { v4 as uuidv4 } from "uuid";
import {
  START,
  END,
  MessagesAnnotation,
  StateGraph,
  MemorySaver,
} from "@langchain/langgraph";
import { SystemMessage } from "@langchain/core/messages";

// Define Models
const llm = new ChatOllama({
  model: "qwen2.5:7b-instruct", // Default value
  temperature: 1
});

// Chat Prompt Template
const promptTemplate = ChatPromptTemplate.fromMessages([
  new SystemMessage("You talk like a pirate. Answer all questions to the best of your ability."),
  new MessagesPlaceholder("messages")
]);

// Define the function that calls the model
const callModel = async (state) => {
  const promptValue = await promptTemplate.invoke({messages: state.messages});
  const response = await llm.invoke(promptValue);
  return { messages: response };
};

// Define a new graph
const workflow = new StateGraph(MessagesAnnotation)
  // Define the node and edge
  .addNode("model", callModel)
  .addEdge(START, "model")
  .addEdge("model", END);

// Add memory
const memory = new MemorySaver();
const app = workflow.compile({ checkpointer: memory });

// Define configuration to distinct each session
const config = { configurable: { thread_id: uuidv4() } };

// Start the chat
const input = [
  {
    role: "user",
    content: "Hi! I'm Bob.",
  },
];

const input2 = [
  {
    role: "user",
    content: "What is my name?",
  },
];

const output = await app.invoke({ messages: input }, config);
const output2 = await app.invoke({ messages: input2 }, config);
console.log(output2.messages[output2.messages.length - 1]);