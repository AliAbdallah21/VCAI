# scripts/test_database.py
"""
Test PostgreSQL database connection and verify schema.

Usage:
    python scripts/test_database.py
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def test_connection():
    """Test basic database connection."""
    print("\n[1/4] Testing database connection...")
    
    try:
        import psycopg2
        
        # Get connection string from environment or use default
        database_url = os.getenv('DATABASE_URL', 'postgresql://postgres:Ali24680#@localhost:5432/vcai')
        
        # Parse the URL manually for psycopg2
        # Format: postgresql://user:password@host:port/database
        conn = psycopg2.connect(database_url)
        conn.close()
        
        print("  ✅ Database connection successful!")
        return True
        
    except Exception as e:
        print(f"  ❌ Connection failed: {str(e)}")
        return False


def test_tables():
    """Verify all tables exist."""
    print("\n[2/4] Checking tables...")
    
    try:
        import psycopg2
        
        database_url = os.getenv('DATABASE_URL', 'postgresql://postgres:Ali24680#@localhost:5432/vcai')
        conn = psycopg2.connect(database_url)
        cur = conn.cursor()
        
        # Get all tables
        cur.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
            ORDER BY table_name;
        """)
        
        tables = [row[0] for row in cur.fetchall()]
        
        expected_tables = [
            'users',
            'personas',
            'sessions',
            'messages',
            'emotion_logs',
            'checkpoints',
            'user_stats'
        ]
        
        for table in expected_tables:
            if table in tables:
                print(f"  ✅ {table}")
            else:
                print(f"  ❌ {table} - MISSING!")
        
        cur.close()
        conn.close()
        
        return all(t in tables for t in expected_tables)
        
    except Exception as e:
        print(f"  ❌ Error: {str(e)}")
        return False


def test_personas():
    """Check personas are loaded."""
    print("\n[3/4] Checking personas...")
    
    try:
        import psycopg2
        
        database_url = os.getenv('DATABASE_URL', 'postgresql://postgres:Ali24680#@localhost:5432/vcai')
        conn = psycopg2.connect(database_url)
        cur = conn.cursor()
        
        cur.execute("SELECT id, name_ar, difficulty FROM personas ORDER BY difficulty;")
        personas = cur.fetchall()
        
        if personas:
            for persona in personas:
                print(f"  ✅ {persona[1]} ({persona[2]})")
        else:
            print("  ❌ No personas found!")
        
        cur.close()
        conn.close()
        
        return len(personas) > 0
        
    except Exception as e:
        print(f"  ❌ Error: {str(e)}")
        return False


def test_sqlalchemy():
    """Test SQLAlchemy connection (what we'll use in the app)."""
    print("\n[4/4] Testing SQLAlchemy connection...")
    
    try:
        from sqlalchemy import create_engine, text
        
        database_url = os.getenv('DATABASE_URL', 'postgresql://postgres:Ali24680#@localhost:5432/vcai')
        engine = create_engine(database_url)
        
        with engine.connect() as conn:
            result = conn.execute(text("SELECT COUNT(*) FROM personas"))
            count = result.scalar()
            print(f"  ✅ SQLAlchemy connected! ({count} personas)")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error: {str(e)}")
        return False


def main():
    print("=" * 60)
    print("VCAI Database Test")
    print("=" * 60)
    
    results = []
    
    results.append(("Connection", test_connection()))
    results.append(("Tables", test_tables()))
    results.append(("Personas", test_personas()))
    results.append(("SQLAlchemy", test_sqlalchemy()))
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {name}: {status}")
    
    passed = sum(1 for _, r in results if r)
    print(f"\nResult: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 Database is ready!")
    else:
        print("\n⚠️ Some tests failed. Check the errors above.")


if __name__ == "__main__":
    main()