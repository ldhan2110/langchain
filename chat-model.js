import 'dotenv/config';
import { Ollama } from "@langchain/ollama";
import { HumanMessage, SystemMessage } from "@langchain/core/messages";
import { ChatPromptTemplate, MessagesPlaceholder } from "@langchain/core/prompts";

// const model = new ChatOpenAI({ model: "gpt-4o-mini", apiKey: process.env.OPENAI_API_KEY });

const model = new Ollama({
  model: "qwen2.5:7b-instruct", // Default value
  temperature: 0,
  maxRetries: 2,
});

const promptTemplate = ChatPromptTemplate.fromMessages([
  new SystemMessage("You talk like a pirate. Answer all questions to the best of your ability."),
  new MessagesPlaceholder("msgs"),
]);

const messages = [
  new HumanMessage("Can you help me?"),
];

const promptValue = await promptTemplate.invoke({msgs: messages});

const response = await model.invoke(promptValue);

console.log(response);