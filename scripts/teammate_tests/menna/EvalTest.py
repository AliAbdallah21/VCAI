"""
Test Example for VCAI Evaluation System
Author: Mena Khaled

This file demonstrates how to use the evaluation system with mock data.
"""

import json
from evaluation import (
    EvaluationManager,
    EvaluationMode,
    ConversationData,
    create_manager,
    evaluate,
    info
)


# ============================================================================
# Mock AI Pipeline (for testing without Ismail's actual implementation)
# ============================================================================

class MockAIPipeline:
    """
    Mock AI Pipeline for testing
    This simulates what Ismail's AI pipeline will return
    """
    
    def analyze_conversation(self, transcript, emotion_log, rag_context):
        """
        Mock analysis - returns sample analysis data
        
        In production, this will be replaced by Ismail's actual AI analysis
        """
        return {
            'topics_discussed': ['price', 'location', 'features', 'payment'],
            'objections_count': 2,
            'closing_signals_count': 1,
            'questions_asked': 5,
            'factual_claims_count': 8,
            'emotional_moments_count': 3,
            'total_turns': len(transcript),
            'rapport_moments': 2
        }
    
    def generate_report(self, analysis, weights, mode):
        """
        Mock report generation - returns sample report
        
        In production, this will be replaced by Ismail's actual AI report generation
        """
        return {
            'overall_score': 78.0,
            'skill_scores': {
                'rapport_building': 85,
                'active_listening': 70,
                'needs_discovery': 75,
                'product_knowledge': 90,
                'objection_handling': 65,
                'emotional_intelligence': 72,
                'closing_skills': 60,
                'communication_clarity': 88
            },
            'checkpoints': {
                'rapport_established': True,
                'needs_identified': True,
                'value_demonstrated': True,
                'objection_handled': True,
                'closing_signal_recognized': False,
                'commitment_achieved': False
            },
            'strengths': [
                'Excellent product knowledge demonstrated',
                'Strong rapport building in opening',
                'Clear and professional communication',
                'Good use of factual information'
            ],
            'areas_to_improve': [
                'Missed closing signal in turn 10',
                'Could improve objection handling timing',
                'Needs to be more assertive in closing',
                'Should ask more probing questions'
            ],
            'turn_feedback': [
                {
                    'turn': 3,
                    'feedback': 'Good question to understand budget',
                    'type': 'positive'
                },
                {
                    'turn': 7,
                    'feedback': 'Objection handling was defensive, try acknowledging concern first',
                    'type': 'improvement'
                },
                {
                    'turn': 10,
                    'feedback': 'Customer showed buying signal, should have asked for commitment',
                    'type': 'critical'
                }
            ],
            'recommended_practice': [
                'Practice closing signal recognition scenarios',
                'Practice objection handling with price concerns',
                'Review best practices for trial close questions'
            ]
        }


# ============================================================================
# Example 1: Basic Usage with Training Mode
# ============================================================================

def example_1_training_mode():
    """Example of evaluating a conversation in training mode"""
    print("\n" + "=" * 60)
    print("EXAMPLE 1: Training Mode Evaluation")
    print("=" * 60)
    
    # Create mock AI pipeline
    ai_pipeline = MockAIPipeline()
    
    # Create evaluation manager
    manager = EvaluationManager(
        ai_pipeline=ai_pipeline,
        mode=EvaluationMode.TRAINING,
        verbose=True
    )
    
    # Create mock conversation data
    conversation = ConversationData(
        session_id="train-001",
        transcript=[
            {'speaker': 'salesperson', 'message': 'السلام عليكم، كيف حالك؟'},
            {'speaker': 'customer', 'message': 'أهلاً، بخير الحمد لله'},
            {'speaker': 'salesperson', 'message': 'ممكن أعرف ميزانيتك للشقة؟'},
            {'speaker': 'customer', 'message': '2 مليون تقريباً'},
            {'speaker': 'salesperson', 'message': 'ممتاز، عندنا شقق في هذا النطاق'},
            {'speaker': 'customer', 'message': 'لكن السعر مرتفع شوية'},
            {'speaker': 'salesperson', 'message': 'فاهم قلقك، بس جودة البناء ممتازة'},
            {'speaker': 'customer', 'message': 'حلو، ممكن أشوف الشقة؟'},
        ],
        emotion_log=[
            {'turn': 1, 'emotion': 'neutral'},
            {'turn': 2, 'emotion': 'neutral'},
            {'turn': 3, 'emotion': 'interested'},
            {'turn': 4, 'emotion': 'interested'},
            {'turn': 5, 'emotion': 'curious'},
            {'turn': 6, 'emotion': 'skeptical'},
            {'turn': 7, 'emotion': 'interested'},
            {'turn': 8, 'emotion': 'excited'},
        ],
        rag_context={
            'property_price': 2500000,
            'property_size': 150,
            'location': 'القاهرة الجديدة',
            'features': ['حمام سباحة', 'جيم', 'أمن']
        },
        customer_persona='medium',
        start_time='2026-02-06T10:00:00',
        end_time='2026-02-06T10:04:32',
        duration_seconds=272,
        mode=EvaluationMode.TRAINING
    )
    
    # Evaluate
    report = manager.evaluate_conversation(conversation)
    
    # Display results
    print("\n" + "=" * 60)
    print("TRAINING MODE RESULTS")
    print("=" * 60)
    print(f"Session ID: {report.session_id}")
    print(f"Overall Score: {report.overall_score}/100")
    print(f"Result: {'PASSED ✓' if report.passed else 'NEEDS IMPROVEMENT ✗'}")
    print(f"Duration: {report.duration_seconds}s ({report.total_turns} turns)")
    print(f"Final Emotion: {report.final_emotion}")
    
    print("\nCheckpoints:")
    for checkpoint, achieved in report.checkpoints.items():
        status = "✓" if achieved else "✗"
        print(f"  {status} {checkpoint.replace('_', ' ').title()}")
    
    print("\nSkill Scores:")
    for skill, score in report.skill_scores.items():
        print(f"  {skill.replace('_', ' ').title():30s} {score}/100")
    
    print("\nTop Strengths:")
    for i, strength in enumerate(report.top_strengths, 1):
        print(f"  {i}. {strength}")
    
    print("\nAreas to Improve:")
    for i, area in enumerate(report.areas_to_improve, 1):
        print(f"  {i}. {area}")
    
    print("\nRecommended Practice:")
    for i, practice in enumerate(report.recommended_practice, 1):
        print(f"  {i}. {practice}")


# ============================================================================
# Example 2: Testing Mode
# ============================================================================

def example_2_testing_mode():
    """Example of evaluating a conversation in testing mode"""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Testing Mode Evaluation")
    print("=" * 60)
    
    # Create mock AI pipeline
    ai_pipeline = MockAIPipeline()
    
    # Create evaluation manager in testing mode
    manager = EvaluationManager(
        ai_pipeline=ai_pipeline,
        mode=EvaluationMode.TESTING,
        verbose=True
    )
    
    # Create mock conversation data
    conversation = ConversationData(
        session_id="test-001",
        transcript=[
            {'speaker': 'salesperson', 'message': 'مرحباً'},
            {'speaker': 'customer', 'message': 'أهلاً'},
            {'speaker': 'salesperson', 'message': 'محتاج شقة؟'},
            {'speaker': 'customer', 'message': 'نعم'},
        ],
        emotion_log=[
            {'turn': 1, 'emotion': 'neutral'},
            {'turn': 2, 'emotion': 'neutral'},
            {'turn': 3, 'emotion': 'interested'},
            {'turn': 4, 'emotion': 'interested'},
        ],
        rag_context={'price': 2000000},
        customer_persona='easy',
        start_time='2026-02-06T11:00:00',
        end_time='2026-02-06T11:02:15',
        duration_seconds=135,
        mode=EvaluationMode.TESTING
    )
    
    # Evaluate
    report = manager.evaluate_conversation(conversation)
    
    # Display results
    print("\n" + "=" * 60)
    print("TESTING MODE RESULTS")
    print("=" * 60)
    print(f"Session ID: {report.session_id}")
    print(f"Overall Score: {report.overall_score}/100")
    print(f"Result: {'PASSED ✓' if report.passed else 'FAILED ✗'}")
    print(f"Pass Threshold: 75/100")
    
    # Note: In testing mode, no turn feedback or practice recommendations (if passed)
    print(f"\nNote: Testing mode provides limited feedback")
    print(f"Turn-by-turn feedback: {'Yes' if report.turn_feedback else 'No (testing mode)'}")


# ============================================================================
# Example 3: Quick Stats
# ============================================================================

def example_3_quick_stats():
    """Example of getting quick stats immediately after call"""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Quick Stats (No AI)")
    print("=" * 60)
    
    # Create manager
    ai_pipeline = MockAIPipeline()
    manager = EvaluationManager(ai_pipeline, EvaluationMode.TRAINING, verbose=False)
    
    # Create conversation data
    conversation = ConversationData(
        session_id="quick-001",
        transcript=[
            {'speaker': 'salesperson', 'message': 'مرحباً'},
            {'speaker': 'customer', 'message': 'أهلاً'},
            {'speaker': 'salesperson', 'message': 'كيف أقدر أساعدك؟'},
            {'speaker': 'customer', 'message': 'محتاج شقة'},
            {'speaker': 'salesperson', 'message': 'تمام، ما ميزانيتك؟'},
            {'speaker': 'customer', 'message': '3 مليون'},
        ],
        emotion_log=[
            {'turn': 6, 'emotion': 'satisfied'}
        ],
        rag_context={},
        customer_persona='medium',
        start_time='2026-02-06T12:00:00',
        end_time='2026-02-06T12:03:20',
        duration_seconds=200,
        mode=EvaluationMode.TRAINING
    )
    
    # Get quick stats (instant, no AI processing)
    stats = manager.get_quick_stats(conversation)
    
    print(f"\nQuick Stats (Instant Feedback):")
    print(f"  Duration: {stats.duration_seconds}s")
    print(f"  Total Turns: {stats.total_turns}")
    print(f"  Final Emotion: {stats.final_emotion}")
    print(f"  Estimated Checkpoints: {stats.checkpoints_estimated}")
    print(f"  Status: {stats.status}")


# ============================================================================
# Example 4: Using Convenience Functions
# ============================================================================

def example_4_convenience_functions():
    """Example using convenience functions"""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Convenience Functions")
    print("=" * 60)
    
    # Create mock AI pipeline
    ai_pipeline = MockAIPipeline()
    
    # Method 1: Using create_manager
    print("\nMethod 1: create_manager()")
    manager = create_manager(ai_pipeline, mode="training", verbose=False)
    print(f"  Manager created: {type(manager).__name__}")
    print(f"  Mode: {manager.get_mode().value}")
    
    # Method 2: Using quick evaluate function
    print("\nMethod 2: evaluate() - one-liner")
    conversation = ConversationData(
        session_id="quick-002",
        transcript=[{'speaker': 'salesperson', 'message': 'test'}],
        emotion_log=[],
        rag_context={},
        customer_persona='easy',
        start_time='2026-02-06T13:00:00',
        end_time='2026-02-06T13:01:00',
        duration_seconds=60,
        mode=EvaluationMode.TRAINING
    )
    
    # One-liner evaluation
    report = evaluate(ai_pipeline, conversation, mode="training")
    print(f"  Report generated: Session {report.session_id}")
    print(f"  Score: {report.overall_score}/100")


# ============================================================================
# Example 5: Report to Dictionary
# ============================================================================

def example_5_serialization():
    """Example of converting report to dictionary for storage"""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Report Serialization")
    print("=" * 60)
    
    ai_pipeline = MockAIPipeline()
    manager = create_manager(ai_pipeline, verbose=False)
    
    conversation = ConversationData(
        session_id="serialize-001",
        transcript=[{'speaker': 'salesperson', 'message': 'test'}],
        emotion_log=[],
        rag_context={},
        customer_persona='easy',
        start_time='2026-02-06T14:00:00',
        end_time='2026-02-06T14:01:00',
        duration_seconds=60,
        mode=EvaluationMode.TRAINING
    )
    
    report = manager.evaluate_conversation(conversation)
    
    # Convert to dictionary for JSON storage
    report_dict = report.to_dict()
    
    print("\nReport as Dictionary:")
    print(json.dumps(report_dict, indent=2, ensure_ascii=False))


# ============================================================================
# Example 6: System Information
# ============================================================================

def example_6_system_info():
    """Example of getting system information"""
    print("\n" + "=" * 60)
    print("EXAMPLE 6: System Information")
    print("=" * 60)
    
    # Display module info
    info()
    
    # Get manager statistics
    ai_pipeline = MockAIPipeline()
    manager = create_manager(ai_pipeline, verbose=False)
    stats = manager.get_statistics()
    
    print("\nManager Statistics:")
    print(json.dumps(stats, indent=2, ensure_ascii=False))


# ============================================================================
# Main Test Runner
# ============================================================================

def main():
    """Run all examples"""
    print("\n" + "=" * 60)
    print("VCAI EVALUATION SYSTEM - TEST EXAMPLES")
    print("=" * 60)
    
    examples = [
        ("Training Mode", example_1_training_mode),
        ("Testing Mode", example_2_testing_mode),
        ("Quick Stats", example_3_quick_stats),
        ("Convenience Functions", example_4_convenience_functions),
        ("Serialization", example_5_serialization),
        ("System Info", example_6_system_info),
    ]
    
    for name, example_func in examples:
        try:
            example_func()
        except Exception as e:
            print(f"\n❌ Error in {name}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
