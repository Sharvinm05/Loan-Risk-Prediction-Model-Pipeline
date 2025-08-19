"""
Security middleware for FastAPI application
"""
import time
import secrets
from fastapi import Request, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi.responses import JSONResponse
from starlette.types import ASGIApp
import logging
from collections import defaultdict
from typing import Dict
import os


# Simple rate limiting store (in production, use Redis or similar)
rate_limit_store: Dict[str, list] = defaultdict(list)

# API key for basic authentication (in production, use proper auth system)
API_KEYS = {
    os.getenv("API_KEY", "demo-key-123456"): "demo-user"
}


class SecurityMiddleware(BaseHTTPMiddleware):
    """Security middleware to add various protections"""
    
    def __init__(self, app: ASGIApp, max_requests: int = 100, window_seconds: int = 3600):
        super().__init__(app)
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        
    async def dispatch(self, request: Request, call_next):
        # Add security headers to response
        response = await call_next(request)
        
        # Security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Content-Security-Policy"] = "default-src 'self'"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware"""
    
    def __init__(self, app: ASGIApp, max_requests: int = 100, window_seconds: int = 3600):
        super().__init__(app)
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        
    async def dispatch(self, request: Request, call_next):
        client_ip = request.client.host
        current_time = time.time()
        
        # Clean old requests
        rate_limit_store[client_ip] = [
            req_time for req_time in rate_limit_store[client_ip]
            if current_time - req_time < self.window_seconds
        ]
        
        # Check rate limit
        if len(rate_limit_store[client_ip]) >= self.max_requests:
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={"detail": "Rate limit exceeded"}
            )
        
        # Add current request
        rate_limit_store[client_ip].append(current_time)
        
        response = await call_next(request)
        return response


class ContentLengthMiddleware(BaseHTTPMiddleware):
    """Middleware to limit request size"""
    
    def __init__(self, app: ASGIApp, max_size: int = 1024 * 1024):  # 1MB default
        super().__init__(app)
        self.max_size = max_size
        
    async def dispatch(self, request: Request, call_next):
        # Check content length
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > self.max_size:
            return JSONResponse(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                content={"detail": "Request too large"}
            )
        
        response = await call_next(request)
        return response


async def verify_api_key(request: Request):
    """Verify API key for authentication"""
    # Skip authentication for health check
    if request.url.path == "/health":
        return None
        
    api_key = request.headers.get("X-API-Key")
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required"
        )
    
    if api_key not in API_KEYS:
        # Log failed authentication attempt
        logging.warning(f"Invalid API key attempt from {request.client.host}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    
    return API_KEYS[api_key]


def setup_security_logging():
    """Setup security-focused logging"""
    security_logger = logging.getLogger("security")
    security_handler = logging.FileHandler("security.log")
    security_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    security_handler.setFormatter(security_formatter)
    security_logger.addHandler(security_handler)
    security_logger.setLevel(logging.INFO)
    
    return security_logger


def validate_model_file_integrity(file_path: str) -> bool:
    """Basic file integrity check for model files"""
    try:
        # Check if file exists and is readable
        if not os.path.exists(file_path):
            logging.error(f"Model file not found: {file_path}")
            return False
            
        # Check file size (models shouldn't be empty or suspiciously large)
        file_size = os.path.getsize(file_path)
        if file_size == 0:
            logging.error(f"Model file is empty: {file_path}")
            return False
            
        if file_size > 1024 * 1024 * 1024:  # 1GB limit
            logging.error(f"Model file suspiciously large: {file_path}")
            return False
            
        return True
        
    except Exception as e:
        logging.error(f"Error validating model file {file_path}: {e}")
        return False