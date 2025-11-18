#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import sys
import math
import argparse
from pathlib import Path
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI()

# Default paths
DEFAULT_EMBEDDING_FILE = "./data/conversations_embedding.json"
DEFAULT_TEXTONLY_FILE = "./data/conversations_textonly.json"

def text_to_embedding(text):
    """Convert input text to embedding using OpenAI API"""
    max_chars = 10000
    if len(text) > max_chars:
        text = text[:max_chars] + "..."
        print(f"Warning: Query text truncated to {max_chars} characters")
    
    try:
        response = client.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error creating embedding for query: {e}")
        return None

def load_embeddings(embedding_file):
    """Load conversation embeddings from file"""
    try:
        with open(embedding_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"Error: File {embedding_file} not found.")
        return None
    except json.JSONDecodeError as e:
        print(f"Error: JSON decode failed: {e}")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None

def load_textonly_conversations(textonly_file):
    """Load text-only conversation data"""
    try:
        with open(textonly_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"Error: File {textonly_file} not found.")
        return None
    except json.JSONDecodeError as e:
        print(f"Error: JSON decode failed: {e}")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None

def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors"""
    if len(vec1) != len(vec2):
        return 0.0
    
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    magnitude_a = math.sqrt(sum(a * a for a in vec1))
    magnitude_b = math.sqrt(sum(b * b for b in vec2))
    
    if magnitude_a == 0.0 or magnitude_b == 0.0:
        return 0.0
    
    return dot_product / (magnitude_a * magnitude_b)

def get_max_index(embedding_file):
    """Get maximum available index"""
    data = load_embeddings(embedding_file)
    if data is None:
        return None
    
    max_index = len(data) - 1
    print(f"Available indexes: 0 ~ {max_index} (Total: {len(data)} conversations)")
    return max_index

def get_conversation_by_index(index, textonly_file):
    """Get conversation by specified index from text-only file"""
    data = load_textonly_conversations(textonly_file)
    if data is None:
        return None
    
    try:
        index = int(index)
    except ValueError:
        print("Error: Index must be an integer.")
        return None
    
    if index < 0 or index >= len(data):
        print(f"Error: Index {index} is out of range. Available range: 0 ~ {len(data) - 1}")
        return None
    
    conversation = data[index]
    print(f"Retrieved conversation #{index}.")
    return conversation

def print_conversation_summary(conversation, index):
    """Print conversation summary"""
    if not conversation:
        return
    
    stored_index = conversation.get('index', 'N/A')
    print(f"\n=== Conversation #{index} ===")
    print(f"Stored Index: {stored_index}")
    print(f"Title: {conversation.get('title', 'N/A')}")
    
    messages = conversation.get('messages', [])
    print(f"Messages: {len(messages)}")
    
    if messages:
        print("\nMessage breakdown:")
        roles = {}
        for msg in messages:
            role = msg.get('role', 'unknown')
            roles[role] = roles.get(role, 0) + 1
        for role, count in roles.items():
            print(f"  {role}: {count}")

def find_similar_conversations(query_text, embedding_file, top_k=5):
    """Find top-k most similar conversations to the query text"""
    print(f"Creating embedding for query: '{query_text[:100]}...'")
    
    # Convert query to embedding
    query_embedding = text_to_embedding(query_text)
    if query_embedding is None:
        return []
    
    print("Loading conversation embeddings...")
    embeddings_data = load_embeddings(embedding_file)
    if embeddings_data is None:
        return []
    
    print(f"Computing similarity with {len(embeddings_data)} conversations...")
    
    # Calculate similarities
    similarities = []
    for conv in embeddings_data:
        conv_embedding = conv.get('embedding', [])
        if conv_embedding:
            similarity = cosine_similarity(query_embedding, conv_embedding)
            similarities.append({
                'index': conv.get('index', -1),
                'title': conv.get('title', 'Untitled'),
                'similarity': similarity,
                'message_count': conv.get('message_count', 0),
                'text_length': conv.get('text_length', 0),
                'text_preview': conv.get('text_preview', '')
            })
    
    # Sort by similarity (descending)
    similarities.sort(key=lambda x: x['similarity'], reverse=True)
    
    # Return top-k results
    return similarities[:top_k]

def print_similarity_results(results, query_text):
    """Print similarity search results"""
    print(f"\nTop {len(results)} similar conversations for: '{query_text[:50]}...'\n")
    print("=" * 80)
    
    for i, result in enumerate(results, 1):
        print(f"{i}. Conversation #{result['index']} (Similarity: {result['similarity']:.4f})")
        print(f"   Title: {result['title']}")
        print(f"   Messages: {result['message_count']}, Text length: {result['text_length']} chars")
        print(f"   Preview: {result['text_preview'][:150]}...")
        print("-" * 80)

def show_conversation_details(index, textonly_file):
    """Show detailed conversation content"""
    conversation = get_conversation_by_index(index, textonly_file)
    if not conversation:
        return
    
    print(f"\n=== Detailed View: Conversation #{index} ===")
    print(f"Title: {conversation.get('title', 'Untitled')}")
    print(f"Index: {conversation.get('index', 'N/A')}")
    
    messages = conversation.get('messages', [])
    print(f"Total Messages: {len(messages)}")
    print("=" * 60)
    
    for i, msg in enumerate(messages, 1):
        role = msg.get('role', 'unknown')
        text = msg.get('text', '')
        print(f"\n[{i}] {role.upper()}:")
        print("-" * 40)
        print(text[:500] + ("..." if len(text) > 500 else ""))

def main():
    parser = argparse.ArgumentParser(
        description="Retrieve and search ChatGPT conversations using embeddings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  max                     - Show maximum available index
  summary <index>         - Show conversation summary
  retrieve <index>        - Get full conversation content
  details <index>         - Show detailed conversation view
  search "<query>"        - Find similar conversations using semantic search

Examples:
  # Using default embedding file
  python retrieval.py max
  python retrieval.py summary 42
  python retrieval.py search "Python programming error"
  
  # Specifying custom embedding file
  python retrieval.py --embedding-file ./custom/path/conversations_embedding.json search "machine learning"
  python retrieval.py -e /path/to/embeddings.json max
        """
    )
    
    parser.add_argument(
        '--embedding-file', '-e',
        default=DEFAULT_EMBEDDING_FILE,
        help=f'Path to conversations_embedding.json (default: {DEFAULT_EMBEDDING_FILE})'
    )
    
    parser.add_argument(
        '--textonly-file', '-t',
        default=DEFAULT_TEXTONLY_FILE,
        help=f'Path to conversations_textonly.json (default: {DEFAULT_TEXTONLY_FILE})'
    )
    
    parser.add_argument(
        'command',
        help='Command to execute (max, summary, retrieve, details, search)'
    )
    
    parser.add_argument(
        'args',
        nargs='*',
        help='Additional arguments for the command'
    )
    
    args = parser.parse_args()
    
    # Validate files exist
    embedding_file = Path(args.embedding_file)
    textonly_file = Path(args.textonly_file)
    
    if not embedding_file.exists():
        print(f"Error: Embedding file not found: {embedding_file}")
        print("Please ensure the file exists or specify correct path with --embedding-file")
        sys.exit(1)
    
    if args.command in ['summary', 'retrieve', 'details'] and not textonly_file.exists():
        print(f"Error: Text-only file not found: {textonly_file}")
        print("Please ensure the file exists or specify correct path with --textonly-file")
        sys.exit(1)
    
    # Execute commands
    if args.command == "max":
        get_max_index(embedding_file)
        
    elif args.command == "summary":
        if not args.args:
            print("Error: Index required for summary command")
            print("Usage: python retrieval.py summary <index>")
            sys.exit(1)
        index = args.args[0]
        conversation = get_conversation_by_index(index, textonly_file)
        if conversation:
            print_conversation_summary(conversation, index)
            
    elif args.command == "retrieve":
        if not args.args:
            print("Error: Index required for retrieve command")
            print("Usage: python retrieval.py retrieve <index>")
            sys.exit(1)
        index = args.args[0]
        conversation = get_conversation_by_index(index, textonly_file)
        if conversation:
            print(json.dumps(conversation, ensure_ascii=False, indent=2))
            
    elif args.command == "details":
        if not args.args:
            print("Error: Index required for details command")
            print("Usage: python retrieval.py details <index>")
            sys.exit(1)
        index = args.args[0]
        show_conversation_details(index, textonly_file)
        
    elif args.command == "search":
        if not args.args:
            print("Error: Query text required for search command")
            print("Usage: python retrieval.py search \"<query text>\"")
            sys.exit(1)
        query_text = " ".join(args.args)
        results = find_similar_conversations(query_text, embedding_file, top_k=5)
        if results:
            print_similarity_results(results, query_text)
        else:
            print("No similar conversations found.")
            
    else:
        print(f"Error: Unknown command '{args.command}'")
        print("Available commands: max, summary, retrieve, details, search")
        sys.exit(1)

if __name__ == "__main__":
    main()