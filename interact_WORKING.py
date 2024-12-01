import os
from openai import OpenAI
from typing import List, Dict, Any, Tuple
import numpy as np
from dataclasses import dataclass
import spacy
import networkx as nx
from sentence_transformers import SentenceTransformer
from datetime import datetime

@dataclass
class ConceptContext:
    """Stores the meaning and context for a concept"""
    original_meaning: str
    current_meaning: str
    embedding: np.ndarray
    last_used: datetime
    clarifications: List[str]
    definition: str
    related_concepts: List[str]

class InteractiveMeaningFramework:
    def __init__(self, api_key: str = None):
        self.client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
        self.nlp = spacy.load("en_core_web_sm")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.concept_store = {}
        self.conversation_history = []
        self.relationship_graph = nx.DiGraph()

    def _get_concept_definition(self, concept: str, context: str) -> Tuple[str, List[str]]:
        """Query LLM to get concept definition and related concepts based on context"""
        messages = [
            {
                "role": "system",
                "content": (
                    "Define the following word in the context provided. "
                    "Return a JSON-like response with two fields: "
                    "'definition' - a clear, contextual definition of the word, "
                    "'related_concepts' - a list of closely related concepts from the context."
                )
            },
            {
                "role": "user",
                "content": f"Word: {concept}\nContext: {context}"
            }
        ]
        
        response = self.client.chat.completions.create(
            messages=messages,
            model="gpt-4o-mini",
            response_format={ "type": "json_object" }
        )
        
        result = eval(response.choices[0].message.content)
        return result['definition'], result['related_concepts']

    def _process_concept(self, concept: str, context: str) -> None:
        """Process a single concept: get definition, embedding, and store"""
        if concept not in self.concept_store:
            # Get definition and related concepts from LLM
            definition, related_concepts = self._get_concept_definition(concept, context)
            
            # Create embedding using definition for more meaningful representation
            embedding = self.embedding_model.encode([definition])[0]
            
            # Store concept with its definition
            self.concept_store[concept] = ConceptContext(
                original_meaning=concept,
                current_meaning=concept,
                embedding=embedding,
                last_used=datetime.now(),
                clarifications=[],
                definition=definition,
                related_concepts=related_concepts
            )
            
            # Add relationships to graph
            for related in related_concepts:
                self.relationship_graph.add_edge(
                    concept,
                    related,
                    relationship="related_to"
                )

    def start_conversation(self, initial_query: str) -> str:
        """Process initial query and start conversation with enhanced concept understanding"""
        # Process the query
        doc = self.nlp(initial_query)
        
        # Extract meaningful concepts (not just tokens)
        concepts = [
            token.text for token in doc 
            if not token.is_stop and not token.is_punct and token.pos_ in ['NOUN', 'VERB', 'ADJ']
        ]
        
        # Process each concept
        for concept in concepts:
            self._process_concept(concept, initial_query)
            
            # Add syntactic relationships
            token = next(t for t in doc if t.text == concept)
            if token.head.text != token.text:
                self.relationship_graph.add_edge(
                    token.head.text,
                    token.text,
                    relationship=token.dep_
                )

        # Generate comprehensive response using concept network
        response = self._generate_response(initial_query)
        
        # Store in conversation history
        self.conversation_history.append({
            "role": "user",
            "content": initial_query,
            "timestamp": datetime.now()
        })
        self.conversation_history.append({
            "role": "assistant",
            "content": response,
            "timestamp": datetime.now()
        })
        
        return response

    def clarify_concept(self, concept: str, clarification: str) -> str:
        """Allow user to clarify the meaning of a concept"""
        if concept in self.concept_store:
            # Update concept with new clarification
            self.concept_store[concept].clarifications.append(clarification)
            self.concept_store[concept].current_meaning = clarification
            self.concept_store[concept].embedding = self.embedding_model.encode([clarification])[0]
            self.concept_store[concept].last_used = datetime.now()
            
            # Generate confirmation and context update
            response = self._generate_response(
                f"Updated understanding of {concept} with clarification: {clarification}. "
                "How does this affect our previous discussion?"
            )
            
            self.conversation_history.append({
                "role": "user",
                "content": f"Clarification for '{concept}': {clarification}",
                "timestamp": datetime.now()
            })
            self.conversation_history.append({
                "role": "assistant",
                "content": response,
                "timestamp": datetime.now()
            })
            
            return response
        else:
            return f"Concept '{concept}' not found in our conversation."

    def continue_conversation(self, user_input: str) -> str:
        """Continue the conversation with new user input"""
        # Process new input
        doc = self.nlp(user_input)
        
        # Extract meaningful concepts
        concepts = [
            token.text for token in doc 
            if not token.is_stop and not token.is_punct and token.pos_ in ['NOUN', 'VERB', 'ADJ']
        ]
        
        # Process any new concepts
        for concept in concepts:
            if concept not in self.concept_store:
                self._process_concept(concept, user_input)
            else:
                self.concept_store[concept].last_used = datetime.now()

        # Generate response considering context
        response = self._generate_response(user_input)
        
        # Update conversation history
        self.conversation_history.append({
            "role": "user",
            "content": user_input,
            "timestamp": datetime.now()
        })
        self.conversation_history.append({
            "role": "assistant",
            "content": response,
            "timestamp": datetime.now()
        })
        
        return response

    def _generate_response(self, query: str) -> str:
        """Generate response using LLM with enhanced context"""
        # Create detailed context string from concept network
        context = self._build_enhanced_context_string()
        
        messages = [
            {
                "role": "system",
                "content": (
                    "You are engaging in a conversation where concepts have specific "
                    "meanings and relationships. Use this concept network to provide "
                    f"detailed, contextually aware responses:\n\n{context}"
                )
            }
        ]
        
        # Add relevant conversation history
        for entry in self.conversation_history[-4:]:
            messages.append({
                "role": entry["role"],
                "content": entry["content"]
            })
            
        # Add current query
        messages.append({
            "role": "user",
            "content": f"Based on the concept network and previous context, respond to: {query}"
        })
        
        response = self.client.chat.completions.create(
            messages=messages,
            model="gpt-4o-mini"
        )
        
        return response.choices[0].message.content

    def _build_enhanced_context_string(self) -> str:
        """Build detailed context string including definitions and relationships"""
        context_parts = []
        for concept, data in self.concept_store.items():
            part = [f"Concept: {concept}"]
            part.append(f"Definition: {data.definition}")
            if data.clarifications:
                part.append(f"Clarifications: {', '.join(data.clarifications)}")
            part.append(f"Related concepts: {', '.join(data.related_concepts)}")
            
            # Add graph relationships
            if concept in self.relationship_graph:
                relationships = [
                    f"{successor} ({self.relationship_graph[concept][successor]['relationship']})"
                    for successor in self.relationship_graph.successors(concept)
                ]
                if relationships:
                    part.append(f"Relationships: {', '.join(relationships)}")
            
            context_parts.append("\n".join(part))
        
        return "\n\n".join(context_parts)

    def show_current_concepts(self) -> Dict[str, Any]:
        """Display current concepts with their definitions and relationships"""
        return {
            concept: {
                "current_meaning": data.current_meaning,
                "definition": data.definition,
                "related_concepts": data.related_concepts,
                "clarifications": data.clarifications,
                "last_used": data.last_used.isoformat()
            }
            for concept, data in self.concept_store.items()
        }

def interactive_session():
    """Run an interactive session with enhanced concept understanding"""
    framework = InteractiveMeaningFramework()
    
    print("Welcome to the Enhanced Interactive Meaning Framework!")
    print("Enter your initial query:")
    initial_query = input("> ")
    
    response = framework.start_conversation(initial_query)
    print("\nAssistant:", response)
    print("\nConcept Network Generated:")
    concepts = framework.show_current_concepts()
    for concept, details in concepts.items():
        print(f"\n{concept}:")
        print(f"  Definition: {details['definition']}")
        print(f"  Related concepts: {', '.join(details['related_concepts'])}")
        if details['clarifications']:
            print(f"  Clarifications: {', '.join(details['clarifications'])}")

    while True:
        print("\nOptions:")
        print("1. Continue conversation")
        print("2. Clarify a concept")
        print("3. Show concept network")
        print("4. Exit")
        
        choice = input("\nChoose an option (1-4): ")
        
        if choice == "1":
            user_input = input("\nYour message: ")
            response = framework.continue_conversation(user_input)
            print("\nAssistant:", response)
            
        elif choice == "2":
            concept = input("\nEnter concept to clarify: ")
            clarification = input("Enter clarification: ")
            response = framework.clarify_concept(concept, clarification)
            print("\nAssistant:", response)
            
        elif choice == "3":
            concepts = framework.show_current_concepts()
            print("\nCurrent Concept Network:")
            for concept, details in concepts.items():
                print(f"\n{concept}:")
                print(f"  Definition: {details['definition']}")
                print(f"  Related concepts: {', '.join(details['related_concepts'])}")
                if details['clarifications']:
                    print(f"  Clarifications: {', '.join(details['clarifications'])}")
                print(f"  Last used: {details['last_used']}")
                
        elif choice == "4":
            print("\nPeace :)")
            break
        else:
            print("\nInvalid choice. Please choose 1-4.")

if __name__ == "__main__":
    interactive_session()
