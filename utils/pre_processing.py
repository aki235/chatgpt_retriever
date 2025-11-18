#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import sys
import argparse
from pathlib import Path
import math
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI()

def text2embedding(text):
    """Convert text to embedding vector using OpenAI API"""
    # Truncate text to avoid token limit (roughly 8000 tokens = ~30000 characters)
    max_chars = 10000
    if len(text) > max_chars:
        text = text[:max_chars] + "..."
        print(f"  Warning: Text truncated to {max_chars} characters")
    
    # return [0.0] * 1536

    try:
        response = client.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"  Error creating embedding: {e}")
        # Return zero vector as fallback
        return [0.0] * 1536  # text-embedding-3-small returns 1536 dimensions

def convert_unicode_escapes(input_file, output_file):
    """Convert Unicode escapes in conversations.json to readable format"""
    print(f"Converting Unicode escapes: {input_file} -> {output_file}")
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"Loaded {len(data)} conversations")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"Unicode conversion completed: {output_file}")
        return True
        
    except Exception as e:
        print(f"Error in Unicode conversion: {e}")
        return False

def add_index_to_conversations(input_file, output_file):
    """Add explicit index numbers to each conversation"""
    print(f"Adding index numbers: {input_file} -> {output_file}")
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            conversations = json.load(f)
        
        print(f"Loaded {len(conversations)} conversations")
        
        # Add index to each conversation
        for i, conversation in enumerate(conversations):
            conversation['index'] = i
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(conversations, f, ensure_ascii=False, indent=2)
        
        print(f"Index addition completed: {output_file}")
        return True
        
    except Exception as e:
        print(f"Error adding indexes: {e}")
        return False

def extract_messages_from_conversation(conversation):
    """Extract chat messages from a conversation's mapping"""
    messages = []
    mapping = conversation.get('mapping', {})
    
    for node_id, node in mapping.items():
        message = node.get('message')
        if message and message.get('content'):
            author = message.get('author', {})
            role = author.get('role', 'unknown')
            content = message.get('content', {})
            parts = content.get('parts', [])
            
            # Join all parts into a single text
            text = '\n'.join(str(part) for part in parts if part)
            
            if text.strip():  # Only include non-empty messages
                messages.append({
                    'role': role,
                    'text': text.strip()
                })
    
    return messages

def extract_text_only(input_file, output_file):
    """Extract only chat text and index from conversations"""
    print(f"Extracting text-only data: {input_file} -> {output_file}")
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            conversations = json.load(f)
        
        print(f"Loaded {len(conversations)} conversations")
        
        text_only_conversations = []
        
        for conversation in conversations:
            index = conversation.get('index', -1)
            title = conversation.get('title', 'Untitled')
            messages = extract_messages_from_conversation(conversation)
            
            if messages:  # Only include conversations with actual messages
                text_only_conversations.append({
                    'index': index,
                    'title': title,
                    'messages': messages
                })
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(text_only_conversations, f, ensure_ascii=False, indent=2)
        
        print(f"Text extraction completed: {len(text_only_conversations)} conversations -> {output_file}")
        return True
        
    except Exception as e:
        print(f"Error in text extraction: {e}")
        return False

def create_conversation_embedding(conversation):
    """Create embedding for a single conversation"""
    index = conversation.get('index', -1)
    title = conversation.get('title', '')
    messages = conversation.get('messages', [])
    
    # Combine all messages into single text
    all_text_parts = []
    
    # Add title
    if title:
        all_text_parts.append(f"Title: {title}")
    
    # Add all messages
    for message in messages:
        role = message.get('role', 'unknown')
        text = message.get('text', '')
        if text:
            all_text_parts.append(f"{role.capitalize()}: {text}")
    
    # Join all text parts
    combined_text = '\n\n'.join(all_text_parts)
    
    # Generate embedding
    embedding = text2embedding(combined_text)
    
    return {
        'index': index,
        'title': title,
        'message_count': len(messages),
        'text_length': len(combined_text),
        'embedding': embedding,
        'text_preview': combined_text[:200] + '...' if len(combined_text) > 200 else combined_text
    }

def create_embeddings(input_file, output_file):
    """Create embeddings for all conversations"""
    print(f"Creating embeddings: {input_file} -> {output_file}")
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            conversations = json.load(f)
        
        print(f"Loaded {len(conversations)} conversations")
        print("Creating embeddings...")
        
        embedding_data = []
        
        for i, conversation in enumerate(conversations):
            # Show progress
            if i % 10 == 0 or i == len(conversations) - 1:
                percentage = (i + 1) / len(conversations) * 100
                print(f"Processing: {i + 1:4d}/{len(conversations)} ({percentage:5.1f}%) - {conversation.get('title', 'Untitled')[:50]}")
            
            embedding_conv = create_conversation_embedding(conversation)
            embedding_data.append(embedding_conv)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(embedding_data, f, ensure_ascii=False, indent=2)
        
        print(f"Embedding creation completed: {len(embedding_data)} conversations -> {output_file}")
        print(f"File size: {Path(output_file).stat().st_size / 1024 / 1024:.1f}MB")
        return True
        
    except Exception as e:
        print(f"Error creating embeddings: {e}")
        return False

def process_conversations_direct(data_path):
    """Process conversations.json directly to create text-only and embedding files"""
    data_path = Path(data_path)
    
    # Check if data path exists
    if not data_path.exists():
        print(f"Error: Path {data_path} does not exist")
        return False
    
    # Find conversations.json
    conversations_json = data_path / "conversations.json"
    if not conversations_json.exists():
        print(f"Error: conversations.json not found in {data_path}")
        return False
    
    print(f"Found conversations.json: {conversations_json}")
    print("=" * 60)
    
    # Define output files
    conversations_textonly = data_path / "conversations_textonly.json"
    conversations_embedding = data_path / "conversations_embedding.json"
    
    try:
        # Load and process conversations.json directly
        print("Step 1: Loading and processing conversations.json...")
        with open(conversations_json, 'r', encoding='utf-8') as f:
            conversations = json.load(f)
        
        print(f"Loaded {len(conversations)} conversations")
        
        # Add index and extract text in one step
        print("Step 2: Adding indexes and extracting text-only data...")
        text_only_conversations = []
        
        for i, conversation in enumerate(conversations):
            # Add index
            conversation['index'] = i
            
            # Extract messages
            title = conversation.get('title', 'Untitled')
            messages = extract_messages_from_conversation(conversation)
            
            if messages:  # Only include conversations with actual messages
                text_only_conversations.append({
                    'index': i,
                    'title': title,
                    'messages': messages
                })
        
        # Save text-only file
        with open(conversations_textonly, 'w', encoding='utf-8') as f:
            json.dump(text_only_conversations, f, ensure_ascii=False, indent=2)
        
        print(f"Text extraction completed: {len(text_only_conversations)} conversations -> {conversations_textonly}")
        
        # Step 3: Create embeddings
        print("\nStep 3: Creating embeddings...")
        if not create_embeddings(conversations_textonly, conversations_embedding):
            return False
        
        print("\n" + "=" * 60)
        print("Processing pipeline completed successfully!")
        print(f"Intermediate file: {conversations_textonly}")
        print(f"Final output: {conversations_embedding}")
        
        # Show final statistics
        try:
            with open(conversations_embedding, 'r', encoding='utf-8') as f:
                final_data = json.load(f)
            print(f"Total conversations processed: {len(final_data)}")
            print(f"Text-only file size: {conversations_textonly.stat().st_size / 1024 / 1024:.1f}MB")
            print(f"Embedding file size: {conversations_embedding.stat().st_size / 1024 / 1024:.1f}MB")
        except:
            pass
        
        return True
        
    except Exception as e:
        print(f"Error in processing: {e}")
        return False

def process_conversations(data_path):
    """Complete processing pipeline from conversations.json to conversations_embedding.json"""
    return process_conversations_direct(data_path)

def main():
    parser = argparse.ArgumentParser(
        description="Process ChatGPT conversations.json to create embeddings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pre_processing.py ./data/
  python pre_processing.py /path/to/chatgpt/export/
  python pre_processing.py ../conversations/

This script will:
1. Find conversations.json in the specified path
2. Process conversations and add index numbers
3. Extract text-only data -> conversations_textonly.json
4. Create embeddings using OpenAI API -> conversations_embedding.json
        """
    )
    
    parser.add_argument(
        'data_path',
        help='Path to directory containing conversations.json'
    )
    
    args = parser.parse_args()
    
    print("ChatGPT Conversations Processing Pipeline")
    print("=" * 60)
    print(f"Target path: {args.data_path}")
    print("Note: This requires OpenAI API access for embedding creation")
    print("")
    
    if process_conversations(args.data_path):
        print("\nProcessing completed successfully!")
        sys.exit(0)
    else:
        print("\nProcessing failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()