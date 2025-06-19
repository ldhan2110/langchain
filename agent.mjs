import 'dotenv/config';
import { TavilySearchResults } from "@langchain/community/tools/tavily_search";
import { END, MemorySaver, StateGraph, START, MessagesAnnotation } from "@langchain/langgraph";
import { ToolNode } from "@langchain/langgraph/prebuilt";
import { HumanMessage } from "@langchain/core/messages";
import { ChatOllama } from '@langchain/ollama';

// Define the tools for the agent to use
const agentTools = [new TavilySearchResults({ maxResults: 3 })];
const toolNode = new ToolNode(agentTools);
const agentModel = new ChatOllama({ model: "qwen2.5:7b-instruct", temperature: 0 }).bindTools(agentTools);

// Initialize memory to persist state between graph runs
const agentCheckpointer = new MemorySaver();

// Define the function that determines whether to continue or not
function shouldContinue({ messages }) {
  const lastMessage = messages[messages.length - 1];
  console.log(messages);
  if (lastMessage.tool_calls?.length) {
    return "tools"; // If the LLM makes a tool call, then we route to the "tools" node
  }
  return END;  // Otherwise, we stop (reply to the user) using the special "__end__" node
}

// Define the function that calls the model
async function callModel({messages}) {
  const response = await agentModel.invoke(messages);
  return { messages: [response] }; // We return a list, because this will get added to the existing list
}


// Define new work flows
const workflow = new StateGraph(MessagesAnnotation)
  .addNode("agent", callModel)
  .addEdge(START, "agent") // __start__ is a special name for the entrypoint
  .addNode("tools", toolNode)
  .addEdge("tools", "agent")
  .addConditionalEdges("agent", shouldContinue);

// Finally, we compile it into a LangChain Runnable.
const app = workflow.compile({memory: agentCheckpointer});

// Use the agent
const finalState = await app.invoke({
  messages: [new HumanMessage("what is the weather in sf ?")],
});
console.log(finalState.messages[finalState.messages.length - 1].content);

const nextState = await app.invoke({
  // Including the messages from the previous run gives the LLM context.
  // This way it knows we're asking about the weather in NY
  messages: [...finalState.messages, new HumanMessage("what about ny")],
});
console.log(nextState.messages[nextState.messages.length - 1].content);