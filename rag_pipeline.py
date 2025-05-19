import os
import streamlit as st
import json
import numpy as np
from openai import OpenAI

class PromptCoachRAG:
    def __init__(self, embeddings_path, api_key=None):
        """
        Initialize the RAG pipeline with embeddings and OpenAI client.
        
        Args:
            embeddings_path: Path to the JSON file containing chunk embeddings
            api_key: OpenAI API key (will use environment variable if not provided)
        """
        # Use provided API key or get from environment
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.client = OpenAI(api_key=self.api_key)
        self.load_embeddings(embeddings_path)
    
    def load_embeddings(self, embeddings_path):
        """Load chunk embeddings from JSON file."""
        with open(embeddings_path, 'r') as f:
            self.chunks = json.load(f)
        print(f"Loaded {len(self.chunks)} chunks with embeddings")
    
    def get_embedding(self, text):
        """
        Get embedding for a text using OpenAI's embedding model.
        
        If API key is provided, calls the OpenAI API.
        Otherwise, uses a placeholder embedding for demonstration.
        """
        if self.api_key and self.api_key != "":
            try:
                response = self.client.embeddings.create(
                    input=text, 
                    model="text-embedding-ada-002"
                )
                return response.data[0].embedding
            except Exception as e:
                print(f"Error getting embedding from OpenAI: {e}")
                # Fall back to placeholder if API call fails
                return np.random.normal(0, 1, 1536).tolist()
        else:
            # For demonstration without API key
            return np.random.normal(0, 1, 1536).tolist()
    
    def cosine_similarity(self, a, b):
        """Calculate cosine similarity between two vectors."""
        a = np.array(a)
        b = np.array(b)
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def retrieve_relevant_chunks(self, query, top_k=3):
        """
        Retrieve the most relevant chunks for a query.
        
        Args:
            query: User's prompt to analyze
            top_k: Number of top chunks to retrieve
            
        Returns:
            List of top_k most relevant chunks
        """
        query_embedding = self.get_embedding(query)
        
        # Calculate similarity scores
        for chunk in self.chunks:
            chunk['similarity'] = self.cosine_similarity(query_embedding, chunk['embedding'])
        
        # Sort chunks by similarity score
        sorted_chunks = sorted(self.chunks, key=lambda x: x['similarity'], reverse=True)
        
        # Return top_k chunks
        return sorted_chunks[:top_k]
    
    def analyze_prompt(self, prompt):
        """
        Analyze a user's prompt against best practices.
        
        Args:
            prompt: User's prompt to analyze
            
        Returns:
            Dictionary with analysis results
        """
        # Retrieve relevant chunks
        relevant_chunks = self.retrieve_relevant_chunks(prompt)
        
        # Extract content from relevant chunks
        context = "\n\n".join([chunk['content'] for chunk in relevant_chunks])
        
        system_message = """
        You are a Prompt Coach that helps users improve their prompts for AI systems.
        Analyze the user's prompt against best practices from the Google prompt engineering guide.
        Provide specific feedback and suggestions for improvement.
        """
        
        user_message = f"""
        Here is the user's prompt:
        "{prompt}"
        
        Here are relevant sections from the Google prompt engineering guide:
        {context}
        
        Please analyze this prompt and provide:
        1. An overall assessment
        2. Specific strengths and weaknesses
        3. A refined version of the prompt
        4. Explanation of changes made
        """
        
        if self.api_key and self.api_key != "":
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4o",  # or "gpt-3.5-turbo" for lower cost
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": user_message}
                    ]
                )
                analysis_text = response.choices[0].message.content
                
                # Parse the analysis text into structured format
                # This is a simplified parsing - in production you might want more robust parsing
                lines = analysis_text.split('\n')
                assessment = ""
                strengths = []
                weaknesses = []
                refined_prompt = ""
                explanation = ""
                
                current_section = None
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    
                    if "assessment" in line.lower() or "overall" in line.lower():
                        current_section = "assessment"
                        assessment = line.split(":", 1)[1].strip() if ":" in line else ""
                        continue
                    elif "strength" in line.lower():
                        current_section = "strengths"
                        continue
                    elif "weakness" in line.lower() or "improvement" in line.lower():
                        current_section = "weaknesses"
                        continue
                    elif "refined" in line.lower() or "improved" in line.lower():
                        current_section = "refined"
                        continue
                    elif "explanation" in line.lower() or "changes" in line.lower():
                        current_section = "explanation"
                        continue
                    
                    if current_section == "assessment" and not assessment:
                        assessment = line
                    elif current_section == "strengths" and line.startswith("-"):
                        strengths.append(line[1:].strip())
                    elif current_section == "weaknesses" and line.startswith("-"):
                        weaknesses.append(line[1:].strip())
                    elif current_section == "refined":
                        refined_prompt += line + " "
                    elif current_section == "explanation":
                        explanation += line + " "
                
                # If parsing failed to extract sections, use the full text
                if not assessment and not strengths and not weaknesses and not refined_prompt and not explanation:
                    return {
                        "original_prompt": prompt,
                        "analysis": analysis_text,
                        "relevant_sections": [{"title": chunk['filename'], "content": chunk['content']} for chunk in relevant_chunks]
                    }
                
                analysis = {
                    "assessment": assessment or "This prompt could be improved based on Google's prompt engineering guide.",
                    "strengths": strengths or ["The prompt provides a basic instruction"],
                    "weaknesses": weaknesses or ["The prompt lacks specificity", "Context could be improved"],
                    "refined_prompt": refined_prompt.strip() or f"Improved version of: {prompt}",
                    "explanation": explanation.strip() or "The refined prompt adds more specificity and context."
                }
            except Exception as e:
                print(f"Error analyzing prompt with OpenAI: {e}")
                # Fall back to placeholder if API call fails
                analysis = self._get_placeholder_analysis(prompt)
        else:
            # For demonstration without API key
            analysis = self._get_placeholder_analysis(prompt)
        
        return {
            "original_prompt": prompt,
            "analysis": analysis,
            "relevant_sections": [{"title": chunk['filename'], "content": chunk['content']} for chunk in relevant_chunks]
        }
    
    def _get_placeholder_analysis(self, prompt):
        """Generate placeholder analysis when API is not available."""
        return {
            "assessment": f"Your prompt '{prompt}' could be improved by adding more specificity and context.",
            "strengths": [
                "Provides a basic instruction",
                "Clear primary intent"
            ],
            "weaknesses": [
                "Lacks specific details about the desired output",
                "Missing context about the target audience or purpose",
                "No format specification for the response"
            ],
            "refined_prompt": f"You are an expert content creator. Write a comprehensive, well-researched blog post about {prompt.replace('Write a blog post about ', '')}. Include 5 key sections with headers, practical examples, and actionable takeaways. Format the response with markdown and optimize it for a technical audience.",
            "explanation": "The refined prompt improves on the original by: 1) Adding a persona for the AI to adopt, 2) Specifying the structure and depth expected, 3) Clarifying the format (markdown), and 4) Defining the target audience. These changes follow Google's best practices for effective prompts."
        }
