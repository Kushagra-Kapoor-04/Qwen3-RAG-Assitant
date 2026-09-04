#!/usr/bin/env python
"""
Qwen3 RAG Assistant
Main application entry point with CLI and Streamlit support.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config.settings import settings
from config.prompts import NO_CONTEXT_RESPONSE


def run_cli():
    """Run the command-line interface."""
    from rich.console import Console
    from rich.panel import Panel
    from rich.markdown import Markdown
    
    console = Console()
    
    console.print(Panel.fit(
        "[bold blue]Qwen3 RAG Assistant[/bold blue]\n"
        "[dim]Retrieval-Augmented Generation with Strict Grounding[/dim]",
        border_style="blue"
    ))
    
    # Initialize service
    console.print("\n[yellow]Initializing...[/yellow]")
    
    try:
        from services.query_service import create_query_service
        query_service = create_query_service()
        
        if not query_service.is_ready():
            console.print(
                "[red]No documents indexed. Please run ingestion first:[/red]\n"
                "  python scripts/ingest.py --source ./data/raw"
            )
            return
        
        console.print("[green]Ready![/green]\n")
        console.print("Enter your questions. Type [bold]'quit'[/bold] to exit.\n")
        
        while True:
            try:
                # Get user input
                question = console.input("[bold cyan]You:[/bold cyan] ").strip()
                
                if not question:
                    continue
                
                if question.lower() in ('quit', 'exit', 'q'):
                    console.print("\n[yellow]Goodbye![/yellow]")
                    break
                
                if question.lower() == 'help':
                    console.print("""
[bold]Commands:[/bold]
  quit/exit/q  - Exit the application
  help         - Show this help message
  stats        - Show session statistics
  clear        - Clear query history
                    """)
                    continue
                
                if question.lower() == 'stats':
                    history = query_service.get_query_history()
                    grounded = sum(1 for q in history if q.get('is_grounded', True))
                    console.print(f"""
[bold]Session Statistics:[/bold]
  Queries: {len(history)}
  Grounded: {grounded}
  Rate: {grounded/len(history)*100:.1f}% (if queries > 0)
                    """)
                    continue
                
                if question.lower() == 'clear':
                    query_service.clear_history()
                    console.print("[green]History cleared.[/green]")
                    continue
                
                # Process query
                console.print()
                with console.status("[bold green]Thinking...", spinner="dots"):
                    result = query_service.query(question)
                
                # Display answer
                console.print("[bold green]Assistant:[/bold green]")
                console.print(Markdown(result.answer))
                
                # Display metadata
                meta_parts = []
                if result.sources:
                    meta_parts.append(f"Sources: {len(result.sources)}")
                meta_parts.append(f"Grounded: {'✓' if result.is_grounded else '✗'}")
                if result.was_regenerated:
                    meta_parts.append("Regenerated")
                meta_parts.append(f"Time: {result.processing_time_ms:.0f}ms")
                
                console.print(f"[dim]{' | '.join(meta_parts)}[/dim]\n")
                
            except KeyboardInterrupt:
                console.print("\n[yellow]Use 'quit' to exit.[/yellow]")
            except Exception as e:
                console.print(f"[red]Error: {str(e)}[/red]")
                
    except ImportError as e:
        console.print(f"[red]Missing dependency: {str(e)}[/red]")
        console.print("Install with: pip install -r requirements.txt")


def run_streamlit():
    """Run the Streamlit web interface."""
    import streamlit as st
    
    st.set_page_config(
        page_title="Qwen3 RAG Assistant",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 Qwen3 RAG Assistant")
    st.caption("Retrieval-Augmented Generation with Strict Grounding")
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "query_service" not in st.session_state:
        st.session_state.query_service = None
        st.session_state.initialized = False
    
    # Sidebar
    with st.sidebar:
        st.header("Settings")
        
        top_k = st.slider(
            "Documents to retrieve",
            min_value=1,
            max_value=20,
            value=5
        )
        
        enable_eval = st.checkbox(
            "Enable hallucination detection",
            value=True
        )
        
        enable_regen = st.checkbox(
            "Enable regeneration",
            value=True
        )
        
        st.divider()
        
        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.rerun()
        
        st.divider()
        st.caption("Qwen3 RAG Assistant v1.0")
    
    # Initialize service
    if not st.session_state.initialized:
        with st.spinner("Initializing..."):
            try:
                from services.query_service import create_query_service
                
                st.session_state.query_service = create_query_service(
                    enable_evaluation=enable_eval,
                    enable_regeneration=enable_regen
                )
                
                if st.session_state.query_service.is_ready():
                    st.session_state.initialized = True
                else:
                    st.error(
                        "No documents indexed. Please run ingestion first:\n"
                        "```\npython scripts/ingest.py --source ./data/raw\n```"
                    )
                    return
            except Exception as e:
                st.error(f"Initialization error: {str(e)}")
                return
    
    # Sync UI settings to service dynamically
    if st.session_state.initialized:
        st.session_state.query_service.enable_evaluation = enable_eval
        st.session_state.query_service.enable_regeneration = enable_regen
    
    # Display chat history
    for message in st.session_state.messages:
        role = message["role"]
        name = "User" if role == "user" else "Assistant"
        avatar = "👤" if role == "user" else "🤖"
        
        with st.chat_message(role, avatar=avatar):
            st.markdown(message["content"])
            if "metadata" in message:
                meta = message["metadata"]
                cols = st.columns(4)
                cols[0].caption(f"Sources: {meta.get('sources', 0)}")
                cols[1].caption(f"Grounded: {'✓' if meta.get('grounded') else '✗'}")
                cols[2].caption(f"Regenerated: {'Yes' if meta.get('regenerated') else 'No'}")
                cols[3].caption(f"Time: {meta.get('time_ms', 0):.0f}ms")
    
    # Chat input
    if prompt := st.chat_input("Ask a question..."):
        # Add user message to state
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)
        
        # Assistant response
        with st.chat_message("assistant", avatar="🤖"):
            try:
                # Step 1: Initialize Stream
                # We use a spinner for the initial wait (retrieval + first token)
                placeholder = st.empty()
                full_content = ""
                final_result = None
                
                stream = st.session_state.query_service.stream_query(
                    prompt,
                    top_k=top_k
                )
                
                try:
                    # Get first item (blocks until retrieval done)
                    with st.spinner("Searching context..."):
                        first_item = next(stream)
                    
                    # Process first item
                    if isinstance(first_item, str):
                        full_content += first_item
                        placeholder.markdown(full_content + "▌")
                    else:
                        final_result = first_item
                    
                    # Process rest of stream
                    for item in stream:
                        if isinstance(item, str):
                            full_content += item
                            placeholder.markdown(full_content + "▌")
                        else:
                            final_result = item
                    
                    # Final display
                    placeholder.markdown(full_content)
                    
                    if final_result:
                        # Display metadata
                        cols = st.columns(4)
                        cols[0].caption(f"Sources: {len(final_result.sources)}")
                        cols[1].caption(f"Grounded: {'✓' if final_result.is_grounded else '✗'}")
                        cols[2].caption(f"Regenerated: {'Yes' if final_result.was_regenerated else 'No'}")
                        cols[3].caption(f"Time: {final_result.processing_time_ms:.0f}ms")
                        
                        # Save to history
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": final_result.answer,
                            "metadata": {
                                "sources": len(final_result.sources),
                                "grounded": final_result.is_grounded,
                                "regenerated": final_result.was_regenerated,
                                "time_ms": final_result.processing_time_ms
                            }
                        })
                    else:
                        # Save partial content if no final result (unexpected)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": full_content
                        })
                        
                except StopIteration:
                    # Empty stream
                    st.error("No response generated.")
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
                # Ensure we save what we have so far to avoid history loss
                if full_content:
                     st.session_state.messages.append({
                        "role": "assistant",
                        "content": full_content
                    })


def main():
    """Main entry point."""
    # Ensure directories exist
    settings.ensure_directories()
    
    # Check if running through streamlit
    if "streamlit" in sys.modules or os.environ.get("STREAMLIT_RUN"):
        run_streamlit()
    else:
        # Check for --streamlit flag
        if len(sys.argv) > 1 and sys.argv[1] == "--streamlit":
            import subprocess
            subprocess.run([
                sys.executable, "-m", "streamlit", "run",
                __file__, "--server.headless", "true"
            ])
        else:
            run_cli()


if __name__ == "__main__":
    main()