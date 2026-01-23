# scripts/check_memory.py
"""
Memory Agent Debug Tool
Check sessions and view full conversation history.

Usage:
    cd C:\VCAI
    python scripts/check_memory.py
"""

import sys
sys.path.insert(0, r'C:\VCAI')

from datetime import datetime
from backend.database import get_db_context
from backend.models.session import Session
from backend.models.user import User
from backend.models.persona import Persona
from memory.agent import get_session_memory


def list_all_sessions():
    """List all sessions with details."""
    print("\n" + "=" * 80)
    print("📋 ALL SESSIONS")
    print("=" * 80)
    
    with get_db_context() as db:
        sessions = (
            db.query(Session)
            .order_by(Session.created_at.desc())
            .all()
        )
        
        if not sessions:
            print("No sessions found.")
            return []
        
        session_list = []
        for i, s in enumerate(sessions, 1):
            # Get user and persona names
            user = db.query(User).filter(User.id == s.user_id).first()
            persona = db.query(Persona).filter(Persona.id == s.persona_id).first()
            
            user_name = user.full_name if user else "Unknown"
            persona_name = persona.name_en if persona else "Unknown"
            
            created = s.created_at.strftime("%Y-%m-%d %H:%M") if s.created_at else "N/A"
            
            print(f"\n[{i}] Session ID: {s.id}")
            print(f"    Status: {s.status} | Turns: {s.turn_count} | Difficulty: {s.difficulty}")
            print(f"    User: {user_name} | Persona: {persona_name}")
            print(f"    Created: {created}")
            
            session_list.append({
                "index": i,
                "id": str(s.id),
                "status": s.status,
                "turns": s.turn_count,
                "user": user_name,
                "persona": persona_name
            })
        
        return session_list


def view_session_memory(session_id: str):
    """View full memory for a session."""
    print("\n" + "=" * 80)
    print(f"🧠 SESSION MEMORY: {session_id}")
    print("=" * 80)
    
    try:
        memory = get_session_memory(session_id)
        
        print(f"\n📊 Summary:")
        print(f"   Total Turns: {memory['total_turns']}")
        print(f"   Messages: {len(memory['recent_messages'])}")
        print(f"   Checkpoints: {len(memory['checkpoints'])}")
        
        # Show checkpoints
        if memory['checkpoints']:
            print(f"\n📝 Checkpoints ({len(memory['checkpoints'])}):")
            print("-" * 60)
            for cp in memory['checkpoints']:
                turn_range = cp.get('turn_range', (0, 0))
                summary = cp.get('summary', 'No summary')
                print(f"   Turns {turn_range[0]}-{turn_range[1]}:")
                print(f"   {summary}")
                
                if cp.get('key_points'):
                    print(f"   Key points: {', '.join(cp['key_points'])}")
                print()
        
        # Show full conversation
        if memory['recent_messages']:
            print(f"\n💬 Full Conversation ({len(memory['recent_messages'])} messages):")
            print("-" * 60)
            
            current_turn = None
            for msg in memory['recent_messages']:
                turn = msg.get('turn', '?')
                speaker = msg.get('speaker', 'unknown')
                text = msg.get('text', '')
                emotion = msg.get('emotion', '')
                
                # Print turn separator
                if turn != current_turn:
                    current_turn = turn
                    print(f"\n--- Turn {turn} ---")
                
                # Speaker emoji
                if speaker == 'salesperson':
                    emoji = "👤"
                    label = "Salesperson"
                else:
                    emoji = "🤖"
                    label = "Customer (VC)"
                
                print(f"{emoji} {label}:")
                print(f"   {text}")
                
                if emotion:
                    print(f"   [Emotion: {emotion}]")
        else:
            print("\n💬 No messages in this session.")
        
        return memory
        
    except Exception as e:
        print(f"\n❌ Error loading session: {e}")
        import traceback
        traceback.print_exc()
        return None


def interactive_menu():
    """Interactive menu for exploring sessions."""
    while True:
        print("\n" + "=" * 80)
        print("🎯 MEMORY AGENT DEBUG TOOL")
        print("=" * 80)
        print("\nOptions:")
        print("  [1] List all sessions")
        print("  [2] View session by ID")
        print("  [3] View session by number (from list)")
        print("  [4] View all sessions with conversations")
        print("  [q] Quit")
        
        choice = input("\nEnter choice: ").strip().lower()
        
        if choice == '1':
            sessions = list_all_sessions()
            
        elif choice == '2':
            session_id = input("Enter session ID: ").strip()
            if session_id:
                view_session_memory(session_id)
            
        elif choice == '3':
            sessions = list_all_sessions()
            if sessions:
                try:
                    num = int(input("\nEnter session number: ").strip())
                    if 1 <= num <= len(sessions):
                        view_session_memory(sessions[num - 1]['id'])
                    else:
                        print("Invalid number.")
                except ValueError:
                    print("Please enter a valid number.")
            
        elif choice == '4':
            sessions = list_all_sessions()
            if sessions:
                for s in sessions:
                    view_session_memory(s['id'])
                    input("\nPress Enter for next session...")
            
        elif choice == 'q':
            print("\nGoodbye! 👋")
            break
        
        else:
            print("Invalid choice. Try again.")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Memory Agent Debug Tool")
    parser.add_argument('--session', '-s', help='Session ID to view directly')
    parser.add_argument('--list', '-l', action='store_true', help='List all sessions')
    parser.add_argument('--all', '-a', action='store_true', help='View all sessions with conversations')
    
    args = parser.parse_args()
    
    if args.session:
        view_session_memory(args.session)
    elif args.list:
        list_all_sessions()
    elif args.all:
        sessions = list_all_sessions()
        for s in sessions:
            view_session_memory(s['id'])
    else:
        # Interactive mode
        interactive_menu()


if __name__ == "__main__":
    main()