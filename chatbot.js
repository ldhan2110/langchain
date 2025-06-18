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
import { SystemMessage, HumanMessage, AIMessage, trimMessages } from "@langchain/core/messages";

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

// Load History Chat Messages
const messages = [
  new HumanMessage("hi! I'm bob"),
  new AIMessage("hi!"),
  new HumanMessage("I like vanilla ice cream"),
  new AIMessage("nice"),
  new HumanMessage("whats 2 + 2"),
  new AIMessage("4"),
  new HumanMessage("thanks"),
  new AIMessage("no problem!"),
  new HumanMessage("having fun?"),
  new AIMessage("yes!"),
];


// Define Trimmer Message - which limit the number of message that models need to remember
const trimmer = trimMessages({
  maxTokens: 11,    // Store maximum 10 messages
  strategy: "last",    // Store the last messages only based on Max Token.
  tokenCounter: (msgs) => msgs.length,    // counting the messages
  includeSystem: false,   //	The SystemMessage is always included if possible.
  allowPartial: false,   // Partial trimming (e.g. cutting a long message in half) is not allowed
  startOn: "human",   // Trimming will start from the first Human message and trim pairs from there.
});

console.log(await trimmer.invoke(messages));

// Define the function that calls the model
const callModel = async (state) => {
  const trimMessages = await trimmer.invoke(state.messages); // Add new chat in history
  const promptValue = await promptTemplate.invoke({messages: trimMessages});
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

// Invoke Chat
const output2 = await app.invoke({ messages: [...messages, new HumanMessage("What is name of mine ?")] }, config);
console.log(output2.messages[output2.messages.length - 1]["content"]);