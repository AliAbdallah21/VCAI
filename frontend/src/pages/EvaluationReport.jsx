import { useState, useEffect, useCallback } from 'react';
import { useParams, useNavigate, Link } from 'react-router-dom';
import { evaluationAPI, sessionsAPI } from '../services/api';
import Layout from '../components/Layout';

export default function EvaluationReport() {
    const { sessionId } = useParams();
    const navigate = useNavigate();

    const [loading, setLoading] = useState(true);
    const [session, setSession] = useState(null);
    const [evaluation, setEvaluation] = useState(null);
    const [quickStats, setQuickStats] = useState(null);
    const [error, setError] = useState(null);

    // Fetch evaluation data
    const fetchEvaluation = useCallback(async () => {
        try {
            // Get session info
            const sessionData = await sessionsAPI.getById(sessionId);
            setSession(sessionData);

            // Try to get existing report first
            try {
                const report = await evaluationAPI.getReport(sessionId);
                setEvaluation(report);
                if (report.quick_stats) {
                    setQuickStats(report.quick_stats);
                }
            } catch (reportErr) {
                // No report exists, check status or trigger
                if (reportErr.response?.status === 404) {
                    // Try to get status
                    const status = await evaluationAPI.getStatus(sessionId);

                    if (status.status === 'not_found') {
                        // Need to trigger evaluation
                        await evaluationAPI.triggerEvaluation(sessionId);
                        setEvaluation({ status: 'pending', progress: 0 });
                    } else {
                        setEvaluation(status);
                    }

                    // Get quick stats while waiting
                    try {
                        const stats = await evaluationAPI.getQuickStats(sessionId);
                        setQuickStats(stats.stats);
                    } catch {
                        // Quick stats optional
                    }
                } else {
                    throw reportErr;
                }
            }
        } catch (err) {
            console.error('Evaluation fetch error:', err);
            setError(err.response?.data?.detail || 'Failed to load evaluation');
        } finally {
            setLoading(false);
        }
    }, [sessionId]);

    // Initial fetch
    useEffect(() => {
        fetchEvaluation();
    }, [fetchEvaluation]);

    // Poll for updates if evaluation is pending/processing
    useEffect(() => {
        if (!evaluation || evaluation.status === 'completed' || evaluation.status === 'failed') {
            return;
        }

        const interval = setInterval(async () => {
            try {
                const report = await evaluationAPI.getReport(sessionId);
                setEvaluation(report);
                if (report.status === 'completed' || report.status === 'failed') {
                    clearInterval(interval);
                }
            } catch {
                // Still processing, check status
                try {
                    const status = await evaluationAPI.getStatus(sessionId);
                    setEvaluation(prev => ({ ...prev, ...status }));
                } catch {
                    // Ignore polling errors
                }
            }
        }, 2000);

        return () => clearInterval(interval);
    }, [evaluation, sessionId]);

    // Format duration
    const formatDuration = (seconds) => {
        if (!seconds) return '0:00';
        const mins = Math.floor(seconds / 60);
        const secs = seconds % 60;
        return `${mins}:${secs.toString().padStart(2, '0')}`;
    };

    // Get score color
    const getScoreColor = (score) => {
        if (score >= 80) return 'text-emerald-600';
        if (score >= 60) return 'text-amber-600';
        return 'text-red-600';
    };

    // Get score background
    const getScoreBg = (score) => {
        if (score >= 80) return 'bg-emerald-100';
        if (score >= 60) return 'bg-amber-100';
        return 'bg-red-100';
    };

    // Emotion emoji map
    const emotionEmoji = {
        'curious': '🤔',
        'interested': '😊',
        'happy': '😄',
        'satisfied': '😌',
        'neutral': '😐',
        'confused': '😕',
        'frustrated': '😤',
        'angry': '😠',
        'skeptical': '🤨',
        'excited': '🤩',
    };

    if (loading) {
        return (
            <Layout>
                <div className="p-8 flex items-center justify-center min-h-[60vh]">
                    <div className="text-center">
                        <div className="w-16 h-16 border-4 border-blue-600 border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
                        <p className="text-slate-500">Loading evaluation...</p>
                    </div>
                </div>
            </Layout>
        );
    }

    if (error) {
        return (
            <Layout>
                <div className="p-8">
                    <div className="max-w-2xl mx-auto text-center">
                        <div className="text-5xl mb-4">❌</div>
                        <h2 className="text-xl font-bold text-slate-800 mb-2">Error Loading Evaluation</h2>
                        <p className="text-slate-500 mb-6">{error}</p>
                        <Link
                            to="/dashboard"
                            className="inline-block px-6 py-3 bg-blue-600 text-white rounded-xl hover:bg-blue-700 transition font-medium"
                        >
                            Back to Dashboard
                        </Link>
                    </div>
                </div>
            </Layout>
        );
    }

    const isProcessing = evaluation?.status === 'pending' || evaluation?.status === 'processing';
    const isCompleted = evaluation?.status === 'completed';
    const isFailed = evaluation?.status === 'failed';

    return (
        <Layout>
            <div className="p-8">
                {/* Header */}
                <div className="mb-8 flex items-center justify-between">
                    <div>
                        <h1 className="text-2xl font-bold text-slate-800">Evaluation Report</h1>
                        <p className="text-slate-500">
                            {session?.persona_name || 'Training Session'} • {session?.difficulty || 'medium'} difficulty
                        </p>
                    </div>
                    <Link
                        to="/dashboard"
                        className="px-4 py-2 text-slate-600 hover:text-slate-800 transition"
                    >
                        ← Back to Dashboard
                    </Link>
                </div>

                {/* Processing State */}
                {isProcessing && (
                    <div className="bg-white rounded-2xl p-8 shadow-sm border border-slate-100 mb-8 text-center">
                        <div className="w-20 h-20 border-4 border-blue-600 border-t-transparent rounded-full animate-spin mx-auto mb-6"></div>
                        <h2 className="text-xl font-bold text-slate-800 mb-2">Evaluating Your Performance...</h2>
                        <p className="text-slate-500 mb-4">
                            Our AI is analyzing your conversation. This usually takes a few moments.
                        </p>
                        {evaluation?.progress > 0 && (
                            <div className="max-w-xs mx-auto">
                                <div className="h-2 bg-slate-200 rounded-full overflow-hidden">
                                    <div
                                        className="h-full bg-blue-600 transition-all duration-500"
                                        style={{ width: `${evaluation.progress}%` }}
                                    />
                                </div>
                                <p className="text-sm text-slate-400 mt-2">{evaluation.progress}% complete</p>
                            </div>
                        )}
                    </div>
                )}

                {/* Failed State */}
                {isFailed && (
                    <div className="bg-red-50 rounded-2xl p-8 shadow-sm border border-red-100 mb-8 text-center">
                        <div className="text-5xl mb-4">⚠️</div>
                        <h2 className="text-xl font-bold text-red-800 mb-2">Evaluation Failed</h2>
                        <p className="text-red-600 mb-4">
                            {evaluation?.error || 'Something went wrong during evaluation.'}
                        </p>
                        <button
                            onClick={() => {
                                setEvaluation({ status: 'pending', progress: 0 });
                                evaluationAPI.triggerEvaluation(sessionId);
                            }}
                            className="px-6 py-3 bg-red-600 text-white rounded-xl hover:bg-red-700 transition font-medium"
                        >
                            Retry Evaluation
                        </button>
                    </div>
                )}

                {/* Quick Stats - Always show if available */}
                {quickStats && (
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
                        <div className="bg-white rounded-2xl p-5 shadow-sm border border-slate-100">
                            <p className="text-slate-500 text-sm">Duration</p>
                            <p className="text-2xl font-bold text-slate-800 mt-1">
                                {formatDuration(quickStats.duration_seconds)}
                            </p>
                        </div>
                        <div className="bg-white rounded-2xl p-5 shadow-sm border border-slate-100">
                            <p className="text-slate-500 text-sm">Total Turns</p>
                            <p className="text-2xl font-bold text-slate-800 mt-1">{quickStats.total_turns}</p>
                        </div>
                        <div className="bg-white rounded-2xl p-5 shadow-sm border border-slate-100">
                            <p className="text-slate-500 text-sm">Your Responses</p>
                            <p className="text-2xl font-bold text-slate-800 mt-1">{quickStats.salesperson_turns}</p>
                        </div>
                        <div className="bg-white rounded-2xl p-5 shadow-sm border border-slate-100">
                            <p className="text-slate-500 text-sm">Customer Turns</p>
                            <p className="text-2xl font-bold text-slate-800 mt-1">{quickStats.customer_turns}</p>
                        </div>
                    </div>
                )}

                {/* Emotion Journey */}
                {quickStats?.emotion_journey?.length > 0 && (
                    <div className="bg-white rounded-2xl p-6 shadow-sm border border-slate-100 mb-8">
                        <h3 className="font-semibold text-slate-800 mb-4">Customer Emotion Journey</h3>
                        <div className="flex items-center gap-2 flex-wrap">
                            {quickStats.emotion_journey.map((emotion, i) => (
                                <div key={i} className="flex items-center">
                                    <div className="px-3 py-2 bg-slate-100 rounded-xl text-center">
                                        <span className="text-2xl">{emotionEmoji[emotion] || '😐'}</span>
                                        <p className="text-xs text-slate-600 mt-1 capitalize">{emotion}</p>
                                    </div>
                                    {i < quickStats.emotion_journey.length - 1 && (
                                        <span className="text-slate-300 mx-1">→</span>
                                    )}
                                </div>
                            ))}
                        </div>
                    </div>
                )}

                {/* Completed Evaluation Results */}
                {isCompleted && evaluation?.report && (
                    <>
                        {/* Overall Score */}
                        <div className="bg-white rounded-2xl p-8 shadow-sm border border-slate-100 mb-8">
                            <div className="flex items-center justify-between">
                                <div>
                                    <h2 className="text-xl font-bold text-slate-800 mb-2">Overall Performance</h2>
                                    <div className="flex items-center gap-3">
                                        {evaluation.passed ? (
                                            <span className="px-3 py-1 bg-emerald-100 text-emerald-700 rounded-full text-sm font-medium">
                                                ✓ Passed
                                            </span>
                                        ) : (
                                            <span className="px-3 py-1 bg-red-100 text-red-700 rounded-full text-sm font-medium">
                                                ✗ Needs Improvement
                                            </span>
                                        )}
                                        <span className="text-slate-500 text-sm">
                                            Pass threshold: {evaluation.report?.score_breakdown?.pass_threshold || 75}%
                                        </span>
                                    </div>
                                </div>
                                <div className={`w-24 h-24 rounded-full flex items-center justify-center ${getScoreBg(evaluation.overall_score)}`}>
                                    <span className={`text-3xl font-bold ${getScoreColor(evaluation.overall_score)}`}>
                                        {evaluation.overall_score}
                                    </span>
                                </div>
                            </div>
                        </div>

                        {/* Score Breakdown */}
                        {evaluation.report?.score_breakdown && (
                            <div className="bg-white rounded-2xl p-6 shadow-sm border border-slate-100 mb-8">
                                <h3 className="font-semibold text-slate-800 mb-4">Score Breakdown</h3>
                                <div className="space-y-4">
                                    {Object.entries(evaluation.report.score_breakdown)
                                        .filter(([key]) => !['overall_score', 'pass_threshold'].includes(key))
                                        .map(([category, score]) => (
                                            <div key={category}>
                                                <div className="flex justify-between text-sm mb-1">
                                                    <span className="text-slate-600 capitalize">
                                                        {category.replace(/_/g, ' ')}
                                                    </span>
                                                    <span className={`font-medium ${getScoreColor(score)}`}>{score}%</span>
                                                </div>
                                                <div className="h-2 bg-slate-100 rounded-full overflow-hidden">
                                                    <div
                                                        className={`h-full transition-all duration-500 ${score >= 80 ? 'bg-emerald-500' :
                                                                score >= 60 ? 'bg-amber-500' : 'bg-red-500'
                                                            }`}
                                                        style={{ width: `${score}%` }}
                                                    />
                                                </div>
                                            </div>
                                        ))}
                                </div>
                            </div>
                        )}

                        {/* Summary */}
                        {evaluation.report?.summary && (
                            <div className="bg-white rounded-2xl p-6 shadow-sm border border-slate-100 mb-8">
                                <h3 className="font-semibold text-slate-800 mb-3">Summary</h3>
                                <p className="text-slate-600 leading-relaxed">{evaluation.report.summary}</p>
                            </div>
                        )}

                        {/* Strengths & Improvements */}
                        <div className="grid md:grid-cols-2 gap-6 mb-8">
                            {evaluation.report?.strengths?.length > 0 && (
                                <div className="bg-emerald-50 rounded-2xl p-6 border border-emerald-100">
                                    <h3 className="font-semibold text-emerald-800 mb-3">💪 Strengths</h3>
                                    <ul className="space-y-2">
                                        {evaluation.report.strengths.map((item, i) => (
                                            <li key={i} className="text-emerald-700 text-sm flex items-start gap-2">
                                                <span className="text-emerald-500">✓</span>
                                                {item}
                                            </li>
                                        ))}
                                    </ul>
                                </div>
                            )}
                            {evaluation.report?.improvements?.length > 0 && (
                                <div className="bg-amber-50 rounded-2xl p-6 border border-amber-100">
                                    <h3 className="font-semibold text-amber-800 mb-3">📈 Areas for Improvement</h3>
                                    <ul className="space-y-2">
                                        {evaluation.report.improvements.map((item, i) => (
                                            <li key={i} className="text-amber-700 text-sm flex items-start gap-2">
                                                <span className="text-amber-500">→</span>
                                                {item}
                                            </li>
                                        ))}
                                    </ul>
                                </div>
                            )}
                        </div>
                    </>
                )}

                {/* Action Buttons */}
                <div className="flex flex-col sm:flex-row gap-4 justify-center">
                    <Link
                        to="/dashboard"
                        className="px-6 py-3 bg-slate-100 text-slate-700 rounded-xl hover:bg-slate-200 transition font-medium text-center"
                    >
                        Back to Dashboard
                    </Link>
                    <Link
                        to="/setup"
                        className="px-6 py-3 bg-gradient-to-r from-blue-600 to-purple-600 text-white rounded-xl hover:opacity-95 transition font-medium text-center"
                    >
                        Start New Session →
                    </Link>
                </div>
            </div>
        </Layout>
    );
}
