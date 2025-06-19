
import 'dotenv/config';
import { z } from "zod";
import { ChatOllama } from "@langchain/ollama";
import { ChatOpenAI } from "@langchain/openai";

// 1. Define schema using zod
const PersonSchema = z.object({
  name: z.string(),
  age: z.number(),
  occupation: z.string(),
  location: z.string(),
});

// 2. Initialize the LLM
const llm = new ChatOpenAI({
  temperature: 0,
  model: "gpt-4o-mini",
});

// 3. Wrap LLM with structured output parser
const structuredLLM = llm.withStructuredOutput(PersonSchema);

// 4. Ask the model to extract structured data
const response = await structuredLLM.invoke("John Doe is a software engineer, aged 32, living in San Francisco.");

console.log(response);