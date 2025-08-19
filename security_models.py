"""
Security models for input validation and API security
"""
from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import Optional, Literal
import hashlib
import re


class LoanApplicationInput(BaseModel):
    """Secure input model for loan prediction requests"""
    
    model_config = ConfigDict(
        extra="forbid",  # Prevent extra fields
        json_schema_extra={
            "example": {
                "loanAmount": 3000,
                "apr": 199,
                "nPaidOff": 0,
                "isFunded": 1,
                "state": "CA",
                "leadCost": 0,
                "payFrequency": "B",
                "originallyScheduledPaymentAmount": 6395.19
            }
        }
    )
    
    # Financial fields with validation
    loanAmount: float = Field(
        ..., 
        gt=0, 
        le=1000000,  # Max loan amount
        description="Loan amount in dollars"
    )
    
    apr: float = Field(
        ..., 
        ge=0, 
        le=999,  # Max APR
        description="Annual Percentage Rate"
    )
    
    nPaidOff: int = Field(
        ..., 
        ge=0, 
        le=100,  # Reasonable max
        description="Number of paid off loans"
    )
    
    isFunded: int = Field(
        ..., 
        ge=0, 
        le=1,
        description="Whether loan is funded (0 or 1)"
    )
    
    # State validation with allowlist
    state: str = Field(
        ..., 
        min_length=2, 
        max_length=2,
        description="US State code"
    )
    
    leadCost: float = Field(
        ..., 
        ge=0, 
        le=10000,  # Reasonable max
        description="Lead cost in dollars"
    )
    
    # Payment frequency with limited options
    payFrequency: Literal["B", "M", "W", "S"] = Field(
        ...,
        description="Payment frequency: B(Biweekly), M(Monthly), W(Weekly), S(Semi-monthly)"
    )
    
    originallyScheduledPaymentAmount: float = Field(
        ..., 
        gt=0, 
        le=100000,  # Reasonable max payment
        description="Originally scheduled payment amount"
    )
    
    @field_validator('state')
    @classmethod
    def validate_state_code(cls, v):
        """Validate US state codes"""
        # List of valid US state codes
        valid_states = {
            'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA',
            'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD',
            'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ',
            'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC',
            'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY',
            'DC'  # Include District of Columbia
        }
        
        if v.upper() not in valid_states:
            raise ValueError(f'Invalid state code: {v}')
        return v.upper()


class PredictionResponse(BaseModel):
    """Secure response model for predictions"""
    loanStatus: str = Field(..., description="Predicted loan status")
    riskCategory: str = Field(..., description="Risk category classification")
    confidence: Optional[float] = Field(None, ge=0, le=1, description="Prediction confidence score")


def mask_sensitive_data(data: dict) -> dict:
    """Mask sensitive financial data for logging"""
    masked_data = data.copy()
    
    # Mask financial amounts (show only first 2 digits)
    if 'loanAmount' in masked_data:
        amount = str(masked_data['loanAmount'])
        masked_data['loanAmount'] = amount[:2] + '*' * (len(amount) - 2) if len(amount) > 2 else '***'
    
    if 'originallyScheduledPaymentAmount' in masked_data:
        amount = str(masked_data['originallyScheduledPaymentAmount'])
        masked_data['originallyScheduledPaymentAmount'] = amount[:2] + '*' * (len(amount) - 2) if len(amount) > 2 else '***'
    
    if 'leadCost' in masked_data:
        amount = str(masked_data['leadCost'])
        masked_data['leadCost'] = amount[:1] + '*' * (len(amount) - 1) if len(amount) > 1 else '***'
    
    return masked_data


def generate_request_id(data: dict) -> str:
    """Generate a unique request ID for tracking without exposing sensitive data"""
    # Use a hash of the data for unique identification
    data_str = str(sorted(data.items()))
    return hashlib.sha256(data_str.encode()).hexdigest()[:16]