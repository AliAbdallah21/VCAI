# shared/exceptions.py
"""
Custom exceptions for VCAI project.
All components should use these exceptions for consistency.
"""

from typing import Optional, Any
from shared.constants import ERROR_CODES


class VCAIException(Exception):
    """Base exception for all VCAI errors"""
    
    def __init__(
        self,
        message: str,
        error_code: int,
        details: Optional[dict] = None
    ):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)
    
    def to_dict(self) -> dict:
        return {
            "error": True,
            "error_code": self.error_code,
            "message": self.message,
            "details": self.details
        }


# ══════════════════════════════════════════════════════════════════════════════
# AUTHENTICATION EXCEPTIONS
# ══════════════════════════════════════════════════════════════════════════════

class AuthException(VCAIException):
    """Base authentication exception"""
    pass


class InvalidCredentialsError(AuthException):
    """Raised when login credentials are invalid"""
    
    def __init__(self, message: str = "Invalid email or password"):
        super().__init__(
            message=message,
            error_code=ERROR_CODES["AUTH_INVALID_CREDENTIALS"]
        )


class TokenExpiredError(AuthException):
    """Raised when JWT token has expired"""
    
    def __init__(self, message: str = "Token has expired"):
        super().__init__(
            message=message,
            error_code=ERROR_CODES["AUTH_TOKEN_EXPIRED"]
        )


class TokenInvalidError(AuthException):
    """Raised when JWT token is invalid"""
    
    def __init__(self, message: str = "Invalid token"):
        super().__init__(
            message=message,
            error_code=ERROR_CODES["AUTH_TOKEN_INVALID"]
        )


class UnauthorizedError(AuthException):
    """Raised when user is not authorized"""
    
    def __init__(self, message: str = "Unauthorized access"):
        super().__init__(
            message=message,
            error_code=ERROR_CODES["AUTH_UNAUTHORIZED"]
        )


# ══════════════════════════════════════════════════════════════════════════════
# USER EXCEPTIONS
# ══════════════════════════════════════════════════════════════════════════════

class UserException(VCAIException):
    """Base user exception"""
    pass


class UserNotFoundError(UserException):
    """Raised when user is not found"""
    
    def __init__(self, user_id: str = None, email: str = None):
        identifier = user_id or email or "unknown"
        super().__init__(
            message=f"User not found: {identifier}",
            error_code=ERROR_CODES["USER_NOT_FOUND"],
            details={"user_id": user_id, "email": email}
        )


class UserAlreadyExistsError(UserException):
    """Raised when user already exists"""
    
    def __init__(self, email: str):
        super().__init__(
            message=f"User with email {email} already exists",
            error_code=ERROR_CODES["USER_ALREADY_EXISTS"],
            details={"email": email}
        )


class UserInvalidDataError(UserException):
    """Raised when user data is invalid"""
    
    def __init__(self, message: str, field: str = None):
        super().__init__(
            message=message,
            error_code=ERROR_CODES["USER_INVALID_DATA"],
            details={"field": field}
        )


# ══════════════════════════════════════════════════════════════════════════════
# SESSION EXCEPTIONS
# ══════════════════════════════════════════════════════════════════════════════

class SessionException(VCAIException):
    """Base session exception"""
    pass


class SessionNotFoundError(SessionException):
    """Raised when session is not found"""
    
    def __init__(self, session_id: str):
        super().__init__(
            message=f"Session not found: {session_id}",
            error_code=ERROR_CODES["SESSION_NOT_FOUND"],
            details={"session_id": session_id}
        )


class SessionAlreadyActiveError(SessionException):
    """Raised when user already has an active session"""
    
    def __init__(self, user_id: str, active_session_id: str):
        super().__init__(
            message=f"User {user_id} already has an active session",
            error_code=ERROR_CODES["SESSION_ALREADY_ACTIVE"],
            details={"user_id": user_id, "active_session_id": active_session_id}
        )


class SessionExpiredError(SessionException):
    """Raised when session has expired"""
    
    def __init__(self, session_id: str):
        super().__init__(
            message=f"Session has expired: {session_id}",
            error_code=ERROR_CODES["SESSION_EXPIRED"],
            details={"session_id": session_id}
        )


class SessionMaxDurationError(SessionException):
    """Raised when session exceeds maximum duration"""
    
    def __init__(self, session_id: str, duration: float, max_duration: float):
        super().__init__(
            message=f"Session exceeded maximum duration of {max_duration}s",
            error_code=ERROR_CODES["SESSION_MAX_DURATION"],
            details={
                "session_id": session_id,
                "duration": duration,
                "max_duration": max_duration
            }
        )


# ══════════════════════════════════════════════════════════════════════════════
# PERSONA EXCEPTIONS
# ══════════════════════════════════════════════════════════════════════════════

class PersonaException(VCAIException):
    """Base persona exception"""
    pass


class PersonaNotFoundError(PersonaException):
    """Raised when persona is not found"""
    
    def __init__(self, persona_id: str):
        super().__init__(
            message=f"Persona not found: {persona_id}",
            error_code=ERROR_CODES["PERSONA_NOT_FOUND"],
            details={"persona_id": persona_id}
        )


# ══════════════════════════════════════════════════════════════════════════════
# SCENARIO EXCEPTIONS
# ══════════════════════════════════════════════════════════════════════════════

class ScenarioException(VCAIException):
    """Base scenario exception"""
    pass


class ScenarioNotFoundError(ScenarioException):
    """Raised when scenario is not found"""
    
    def __init__(self, scenario_id: str):
        super().__init__(
            message=f"Scenario not found: {scenario_id}",
            error_code=ERROR_CODES["SCENARIO_NOT_FOUND"],
            details={"scenario_id": scenario_id}
        )


# ══════════════════════════════════════════════════════════════════════════════
# AUDIO EXCEPTIONS
# ══════════════════════════════════════════════════════════════════════════════

class AudioException(VCAIException):
    """Base audio exception"""
    pass


class AudioInvalidFormatError(AudioException):
    """Raised when audio format is invalid"""
    
    def __init__(self, expected_format: str, received_format: str = None):
        super().__init__(
            message=f"Invalid audio format. Expected: {expected_format}",
            error_code=ERROR_CODES["AUDIO_INVALID_FORMAT"],
            details={
                "expected_format": expected_format,
                "received_format": received_format
            }
        )


class AudioTooShortError(AudioException):
    """Raised when audio is too short"""
    
    def __init__(self, duration: float, min_duration: float):
        super().__init__(
            message=f"Audio too short: {duration:.2f}s (minimum: {min_duration}s)",
            error_code=ERROR_CODES["AUDIO_TOO_SHORT"],
            details={"duration": duration, "min_duration": min_duration}
        )


class AudioTooLongError(AudioException):
    """Raised when audio is too long"""
    
    def __init__(self, duration: float, max_duration: float):
        super().__init__(
            message=f"Audio too long: {duration:.2f}s (maximum: {max_duration}s)",
            error_code=ERROR_CODES["AUDIO_TOO_LONG"],
            details={"duration": duration, "max_duration": max_duration}
        )


class AudioProcessingError(AudioException):
    """Raised when audio processing fails"""
    
    def __init__(self, message: str, original_error: Exception = None):
        super().__init__(
            message=f"Audio processing failed: {message}",
            error_code=ERROR_CODES["AUDIO_PROCESSING_FAILED"],
            details={"original_error": str(original_error) if original_error else None}
        )


# ══════════════════════════════════════════════════════════════════════════════
# STT EXCEPTIONS
# ══════════════════════════════════════════════════════════════════════════════

class STTException(VCAIException):
    """Base STT exception"""
    pass


class STTTranscriptionError(STTException):
    """Raised when transcription fails"""
    
    def __init__(self, message: str, original_error: Exception = None):
        super().__init__(
            message=f"Transcription failed: {message}",
            error_code=ERROR_CODES["STT_TRANSCRIPTION_FAILED"],
            details={"original_error": str(original_error) if original_error else None}
        )


class STTModelNotLoadedError(STTException):
    """Raised when STT model is not loaded"""
    
    def __init__(self):
        super().__init__(
            message="STT model is not loaded",
            error_code=ERROR_CODES["STT_MODEL_NOT_LOADED"]
        )


# ══════════════════════════════════════════════════════════════════════════════
# EMOTION EXCEPTIONS
# ══════════════════════════════════════════════════════════════════════════════

class EmotionException(VCAIException):
    """Base emotion exception"""
    pass


class EmotionDetectionError(EmotionException):
    """Raised when emotion detection fails"""
    
    def __init__(self, message: str, original_error: Exception = None):
        super().__init__(
            message=f"Emotion detection failed: {message}",
            error_code=ERROR_CODES["EMOTION_DETECTION_FAILED"],
            details={"original_error": str(original_error) if original_error else None}
        )


# ══════════════════════════════════════════════════════════════════════════════
# LLM EXCEPTIONS
# ══════════════════════════════════════════════════════════════════════════════

class LLMException(VCAIException):
    """Base LLM exception"""
    pass


class LLMGenerationError(LLMException):
    """Raised when LLM generation fails"""
    
    def __init__(self, message: str, original_error: Exception = None):
        super().__init__(
            message=f"LLM generation failed: {message}",
            error_code=ERROR_CODES["LLM_GENERATION_FAILED"],
            details={"original_error": str(original_error) if original_error else None}
        )


class LLMModelNotLoadedError(LLMException):
    """Raised when LLM model is not loaded"""
    
    def __init__(self):
        super().__init__(
            message="LLM model is not loaded",
            error_code=ERROR_CODES["LLM_MODEL_NOT_LOADED"]
        )


# ══════════════════════════════════════════════════════════════════════════════
# TTS EXCEPTIONS
# ══════════════════════════════════════════════════════════════════════════════

class TTSException(VCAIException):
    """Base TTS exception"""
    pass


class TTSSynthesisError(TTSException):
    """Raised when TTS synthesis fails"""
    
    def __init__(self, message: str, original_error: Exception = None):
        super().__init__(
            message=f"TTS synthesis failed: {message}",
            error_code=ERROR_CODES["TTS_SYNTHESIS_FAILED"],
            details={"original_error": str(original_error) if original_error else None}
        )


class TTSVoiceNotFoundError(TTSException):
    """Raised when TTS voice is not found"""
    
    def __init__(self, voice_id: str):
        super().__init__(
            message=f"TTS voice not found: {voice_id}",
            error_code=ERROR_CODES["TTS_VOICE_NOT_FOUND"],
            details={"voice_id": voice_id}
        )


# ══════════════════════════════════════════════════════════════════════════════
# RAG EXCEPTIONS
# ══════════════════════════════════════════════════════════════════════════════

class RAGException(VCAIException):
    """Base RAG exception"""
    pass


class RAGRetrievalError(RAGException):
    """Raised when RAG retrieval fails"""
    
    def __init__(self, message: str, query: str = None, original_error: Exception = None):
        super().__init__(
            message=f"RAG retrieval failed: {message}",
            error_code=ERROR_CODES["RAG_RETRIEVAL_FAILED"],
            details={
                "query": query,
                "original_error": str(original_error) if original_error else None
            }
        )


class RAGIndexNotFoundError(RAGException):
    """Raised when RAG index is not found"""
    
    def __init__(self, index_name: str = None):
        super().__init__(
            message=f"RAG index not found: {index_name or 'default'}",
            error_code=ERROR_CODES["RAG_INDEX_NOT_FOUND"],
            details={"index_name": index_name}
        )


# ══════════════════════════════════════════════════════════════════════════════
# WEBSOCKET EXCEPTIONS
# ══════════════════════════════════════════════════════════════════════════════

class WebSocketException(VCAIException):
    """Base WebSocket exception"""
    pass


class WebSocketConnectionError(WebSocketException):
    """Raised when WebSocket connection fails"""
    
    def __init__(self, message: str, original_error: Exception = None):
        super().__init__(
            message=f"WebSocket connection failed: {message}",
            error_code=ERROR_CODES["WS_CONNECTION_FAILED"],
            details={"original_error": str(original_error) if original_error else None}
        )


class WebSocketMessageError(WebSocketException):
    """Raised when WebSocket message is invalid"""
    
    def __init__(self, message: str, message_type: str = None):
        super().__init__(
            message=f"Invalid WebSocket message: {message}",
            error_code=ERROR_CODES["WS_MESSAGE_INVALID"],
            details={"message_type": message_type}
        )


class WebSocketSessionError(WebSocketException):
    """Raised when WebSocket session is not found"""
    
    def __init__(self, session_id: str):
        super().__init__(
            message=f"WebSocket session not found: {session_id}",
            error_code=ERROR_CODES["WS_SESSION_NOT_FOUND"],
            details={"session_id": session_id}
        )