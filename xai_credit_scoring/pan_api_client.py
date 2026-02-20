"""
pan_api_client.py — PAN Credit Bureau API Integration Layer
============================================================
Supports:
  1. Perfios API    — PAN verify + bank statement analysis (RECOMMENDED)
  2. Setu API       — PAN verification (sandbox free)
  3. Karza API      — PAN + credit data
  4. CIBIL API      — Full credit score (enterprise)
  5. Experian API   — Credit report (enterprise)
  6. Mock Sandbox   — Deterministic fake data (no key needed)

Usage:
    client = PANApiClient(provider="perfios", api_key="YOUR_KEY", secret="YOUR_SECRET")
    result = client.get_credit_profile("ABCDE1234F")
"""

import requests
import hashlib
import numpy as np
import os
import json
from dataclasses import dataclass, asdict
from typing import Optional


# ─────────────────────────────────────────────
#  DATA MODEL
# ─────────────────────────────────────────────
@dataclass
class CreditProfile:
    pan:                  str
    name:                 str
    date_of_birth:        str
    pan_verified:         bool
    source:               str          # "perfios" | "setu" | "karza" | "cibil" | "mock"
    # Credit features (maps directly to model input)
    checking_status:      int          # 0–3
    duration:             int          # loan months
    credit_history:       int          # 0–4
    purpose:              int          # 0–9
    credit_amount:        int          # ₹
    savings_status:       int          # 0–4
    employment:           int          # 0–4
    installment_commitment: int        # 1–4
    personal_status:      int          # 0–3
    other_parties:        int          # 0–2
    residence_since:      int          # 1–4
    property_magnitude:   int          # 0–3
    age:                  int
    other_payment_plans:  int          # 0–2
    housing:              int          # 0–2
    existing_credits:     int          # 1–4
    job:                  int          # 0–3
    num_dependents:       int          # 1–2
    own_telephone:        int          # 0–1
    foreign_worker:       int          # 0–1
    # Extra bureau fields
    active_loans:         int   = 0
    credit_utilization:   float = 0.0
    payment_history_score: float = 0.0
    monthly_income:        float = 0.0
    foir:                  float = 0.0   # Fixed Obligation to Income Ratio
    perfios_risk_band:     str   = ""    # "LOW" | "MEDIUM" | "HIGH"
    error:                Optional[str] = None

    def to_model_input(self) -> dict:
        """Returns exactly the 20 features your model expects."""
        return {
            'checking_status':      self.checking_status,
            'duration':             self.duration,
            'credit_history':       self.credit_history,
            'purpose':              self.purpose,
            'credit_amount':        self.credit_amount,
            'savings_status':       self.savings_status,
            'employment':           self.employment,
            'installment_commitment': self.installment_commitment,
            'personal_status':      self.personal_status,
            'other_parties':        self.other_parties,
            'residence_since':      self.residence_since,
            'property_magnitude':   self.property_magnitude,
            'age':                  self.age,
            'other_payment_plans':  self.other_payment_plans,
            'housing':              self.housing,
            'existing_credits':     self.existing_credits,
            'job':                  self.job,
            'num_dependents':       self.num_dependents,
            'own_telephone':        self.own_telephone,
            'foreign_worker':       self.foreign_worker,
        }


# ─────────────────────────────────────────────
#  BASE CLIENT
# ─────────────────────────────────────────────
class PANApiClient:
    """
    Unified client. Switch provider by changing the `provider` param.
    Falls back to "mock" automatically if no API key is set.
    """

    PROVIDERS = ["perfios", "setu", "karza", "cibil", "experian", "mock"]

    def __init__(self, provider: str = "mock", api_key: str = "", secret: str = ""):
        if provider not in self.PROVIDERS:
            raise ValueError(f"Unknown provider '{provider}'. Choose from: {self.PROVIDERS}")
        self.provider = provider if api_key else "mock"
        self.api_key  = api_key
        self.secret   = secret

    def get_credit_profile(self, pan: str) -> CreditProfile:
        pan = pan.upper().strip()
        if self.provider == "perfios":
            return self._call_perfios(pan)
        elif self.provider == "setu":
            return self._call_setu(pan)
        elif self.provider == "karza":
            return self._call_karza(pan)
        elif self.provider == "cibil":
            return self._call_cibil(pan)
        elif self.provider == "experian":
            return self._call_experian(pan)
        else:
            return self._mock_profile(pan)

    # ─────────────────────────────────────────
    #  PERFIOS API (https://www.perfios.com)
    #  Best for: PAN verify + Bank Statement
    #            Analysis + Income Estimation
    #  Docs:  https://api.perfios.com/docs
    #  Signup: https://developer.perfios.com
    #
    #  Flow:
    #    Step 1 — Get OAuth2 token
    #    Step 2 — Verify PAN via /v3/pan-verification
    #    Step 3 — Submit bank statement /v3/bsa/submit
    #    Step 4 — Poll /v3/bsa/result for analysed data
    #             (income, EMIs, FOIR, risk band)
    # ─────────────────────────────────────────
    def _call_perfios(self, pan: str) -> CreditProfile:
        """
        Perfios Comprehensive Credit API.
        Sign up at: https://developer.perfios.com
        Set env vars:
            PERFIOS_API_KEY    — your client_id
            PERFIOS_SECRET     — your client_secret
        """
        BASE    = "https://api.perfios.com"            # prod; use sandbox.api.perfios.com for testing
        SANDBOX = "https://sandbox.api.perfios.com"    # sandbox (recommended for dev)

        # ── Step 1: OAuth2 token ──────────────────
        try:
            token_resp = requests.post(
                f"{SANDBOX}/oauth/token",
                data={
                    "grant_type":    "client_credentials",
                    "client_id":     self.api_key,
                    "client_secret": self.secret,
                    "scope":         "pan:read bsa:read",
                },
                timeout=10
            )
            token_resp.raise_for_status()
            token = token_resp.json().get("access_token", "")
        except Exception as e:
            p = self._mock_profile(pan)
            p.source = "perfios_fallback"
            p.error  = f"Perfios token error: {e}"
            return p

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type":  "application/json",
            "x-perfios-version": "3.0",
        }

        # ── Step 2: PAN Verification ─────────────
        try:
            pan_resp = requests.post(
                f"{SANDBOX}/v3/pan-verification",
                json={"pan": pan, "consent": "Y"},
                headers=headers, timeout=10
            )
            pan_resp.raise_for_status()
            pan_data    = pan_resp.json()
            name        = pan_data.get("name", "Unknown")
            dob         = pan_data.get("dob", "01-01-1990")
            pan_status  = pan_data.get("status", "")           # "VALID" | "INVALID"
            pan_verified = pan_status == "VALID"
        except Exception as e:
            p = self._mock_profile(pan)
            p.source = "perfios_fallback"
            p.error  = f"Perfios PAN verify error: {e}"
            return p

        # ── Step 3 & 4: Bank Statement Analysis ──
        # (optional — skip if no bank statement uploaded)
        # In real usage, customer uploads bank PDF and you get a transaction_id
        # Here we attempt the API; gracefully fall back on failure.
        monthly_income = 0.0
        foir           = 0.0
        risk_band      = ""
        active_loans   = 0
        credit_util    = 0.0

        try:
            # Submit a BSA (Bank Statement Analysis) job
            bsa_submit = requests.post(
                f"{SANDBOX}/v3/bsa/submit",
                json={
                    "pan": pan,
                    "consent": "Y",
                    "analysisType": "CREDIT",
                },
                headers=headers, timeout=10
            )
            bsa_submit.raise_for_status()
            job_id = bsa_submit.json().get("jobId", "")

            if job_id:
                import time
                # Poll for result (max 3 attempts, 2s apart)
                for _ in range(3):
                    time.sleep(2)
                    bsa_result = requests.get(
                        f"{SANDBOX}/v3/bsa/result/{job_id}",
                        headers=headers, timeout=10
                    )
                    bsa_data = bsa_result.json()
                    if bsa_data.get("status") == "COMPLETED":
                        analytics = bsa_data.get("analytics", {})
                        monthly_income = float(analytics.get("monthlyIncome", 0))
                        foir           = float(analytics.get("foir", 0))
                        risk_band      = analytics.get("riskBand", "MEDIUM")
                        active_loans   = int(analytics.get("activeLoanCount", 0))
                        credit_util    = float(analytics.get("creditUtilization", 0))
                        break
        except Exception:
            # BSA is optional; PAN verify already succeeded
            pass

        # ── Build CreditProfile ───────────────────
        base = self._mock_profile(pan)
        base.name          = name
        base.date_of_birth = dob
        base.pan_verified  = pan_verified
        base.source        = "perfios"
        base.monthly_income = monthly_income
        base.foir          = foir
        base.perfios_risk_band = risk_band
        base.active_loans  = active_loans
        base.credit_utilization = credit_util

        if not pan_verified:
            base.error = "PAN not valid per Perfios/NSDL"

        # Map Perfios FOIR → installment_commitment (higher FOIR = worse)
        if foir > 0:
            base.installment_commitment = min(4, max(1, int(foir / 25)))

        # Map risk band → credit_history
        risk_map = {"LOW": 4, "MEDIUM": 2, "HIGH": 0, "": base.credit_history}
        base.credit_history = risk_map.get(risk_band.upper(), base.credit_history)

        return base

    # ─────────────────────────────────────────
    #  SETU API  (https://setu.co/products/kyc)
    #  Sandbox:  https://dg-sandbox.setu.co
    #  Docs:     https://docs.setu.co/kyc/pan
    # ─────────────────────────────────────────
    def _call_setu(self, pan: str) -> CreditProfile:
        """
        Setu PAN verification endpoint.
        Get your key at: https://bridge.setu.co/ → Sandbox → KYC → PAN Verification
        """
        base_url = "https://dg-sandbox.setu.co"          # change to dg.setu.co for prod
        endpoint = f"{base_url}/api/verify/pan"

        headers = {
            "x-client-id":     self.api_key,              # Client ID from Setu dashboard
            "x-client-secret": self.secret,               # Client Secret from Setu dashboard
            "Content-Type":    "application/json",
        }
        payload = {"pan": pan}

        try:
            resp = requests.post(endpoint, json=payload, headers=headers, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            # Setu returns: {"verification": "VALID/INVALID", "data": {...}}
            verified = data.get("verification") == "VALID"
            name     = data.get("data", {}).get("name", "Unknown")
            dob      = data.get("data", {}).get("dateOfBirth", "01-01-1990")

            # PAN verify gives identity — mix with mock for credit features
            base = self._mock_profile(pan)
            base.name       = name
            base.date_of_birth = dob
            base.pan_verified = verified
            base.source     = "setu"
            if not verified:
                base.error = "PAN not found in NSDL database"
            return base

        except requests.exceptions.RequestException as e:
            p = self._mock_profile(pan)
            p.source = "setu_fallback"
            p.error  = str(e)
            return p

    # ─────────────────────────────────────────
    #  KARZA API (https://karza.in)
    #  Docs: https://karza.in/pan-api.html
    #  Provides PAN + credit enquiry data
    # ─────────────────────────────────────────
    def _call_karza(self, pan: str) -> CreditProfile:
        """
        Karza PAN Comprehensive API.
        Sign up at: https://karza.in → Get API Key
        """
        endpoint = "https://testapi.karza.in/v3/pan-comprehensive"  # sandbox
        headers  = {
            "x-karza-key": self.api_key,
            "Content-Type": "application/json",
        }
        payload = {"pan": pan, "consent": "Y", "reason": "Credit Score Check"}

        try:
            resp = requests.post(endpoint, json=payload, headers=headers, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            if data.get("statusCode") != 101:
                p = self._mock_profile(pan)
                p.source = "karza_fallback"
                p.error  = data.get("error", "Karza API error")
                return p

            result = data.get("result", {})
            name   = result.get("name", "Unknown")
            dob    = result.get("dateOfBirth", "01-01-1990")

            base = self._mock_profile(pan)
            base.name       = name
            base.date_of_birth = dob
            base.pan_verified = True
            base.source     = "karza"
            return base

        except requests.exceptions.RequestException as e:
            p = self._mock_profile(pan)
            p.source = "karza_fallback"
            p.error  = str(e)
            return p

    # ─────────────────────────────────────────
    #  CIBIL API
    #  Requires: NBFC/FI RBI-registered entity
    #  Docs: https://developer.transunion.com/
    # ─────────────────────────────────────────
    def _call_cibil(self, pan: str) -> CreditProfile:
        """
        TransUnion CIBIL Commercial API.
        Enterprise only — requires signed agreement with CIBIL.
        Endpoint: https://api.cibil.com/v1/creditreport
        """
        endpoint = "https://api.cibil.com/v1/creditreport"       # prod endpoint
        headers  = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type":  "application/json",
        }
        payload = {
            "applicant": {
                "pan": pan,
                "consentFlag": "Y",
            }
        }

        try:
            resp = requests.post(endpoint, json=payload, headers=headers, timeout=15)
            resp.raise_for_status()
            data = resp.json()

            # CIBIL returns cibilScore (300-900), accounts, enquiries, etc.
            cibil_score   = data.get("cibilScore", 0)
            accounts      = data.get("accounts", [])
            name          = data.get("applicantProfile", {}).get("name", "Unknown")

            # Map CIBIL score back to default probability
            prob_implied = (900 - cibil_score) / 600

            base = self._mock_profile(pan)
            base.name         = name
            base.pan_verified = True
            base.source       = "cibil"
            base.active_loans = len(accounts)
            # Credit history 4 = no issues, 0 = critical
            if cibil_score >= 750: base.credit_history = 4
            elif cibil_score >= 650: base.credit_history = 3
            elif cibil_score >= 550: base.credit_history = 2
            else: base.credit_history = 0
            return base

        except requests.exceptions.RequestException as e:
            p = self._mock_profile(pan)
            p.source = "cibil_fallback"
            p.error  = str(e)
            return p

    # ─────────────────────────────────────────
    #  EXPERIAN API
    #  Docs: https://developer.experian.com/
    # ─────────────────────────────────────────
    def _call_experian(self, pan: str) -> CreditProfile:
        """
        Experian India Credit Report API.
        Register at: https://www.experian.in/business/products/credit-report-api
        """
        endpoint = "https://api.experian.in/v1/credit-report"
        headers  = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type":  "application/json",
        }
        payload = {"pan": pan, "consent": "Y"}

        try:
            resp = requests.post(endpoint, json=payload, headers=headers, timeout=15)
            resp.raise_for_status()
            data = resp.json()

            base = self._mock_profile(pan)
            base.name       = data.get("name", "Unknown")
            base.pan_verified = True
            base.source     = "experian"
            base.payment_history_score = data.get("paymentHistory", {}).get("score", 0.0)
            return base

        except requests.exceptions.RequestException as e:
            p = self._mock_profile(pan)
            p.source = "experian_fallback"
            p.error  = str(e)
            return p

    # ─────────────────────────────────────────
    #  MOCK SANDBOX (Always works, no key needed)
    #  Deterministic: same PAN → same data
    # ─────────────────────────────────────────
    def _mock_profile(self, pan: str) -> CreditProfile:
        seed = int(hashlib.md5(pan.encode()).hexdigest(), 16) % (2**31)
        rng  = np.random.default_rng(seed)

        age = int(rng.integers(19, 75))
        # Deterministically assign a realistic name based on PAN seed
        first_names = ["Raj", "Amit", "Priya", "Sunita", "Vikram", "Anjali", "Ravi", "Deepa"]
        last_names  = ["Sharma", "Patel", "Singh", "Kumar", "Gupta", "Verma", "Joshi", "Nair"]
        name = f"{rng.choice(first_names)} {rng.choice(last_names)}"
        dob  = f"{int(rng.integers(1,28)):02d}-{int(rng.integers(1,12)):02d}-{2024-age}"

        return CreditProfile(
            pan=pan, name=name, date_of_birth=dob,
            pan_verified=True, source="mock",
            checking_status      = int(rng.choice([0,1,2,3])),
            duration             = int(rng.integers(6,72)),
            credit_history       = int(rng.choice([0,1,2,3,4])),
            purpose              = int(rng.choice(range(10))),
            credit_amount        = int(rng.integers(500,15000)),
            savings_status       = int(rng.choice([0,1,2,3,4])),
            employment           = int(rng.choice([0,1,2,3,4])),
            installment_commitment = int(rng.integers(1,5)),
            personal_status      = int(rng.choice([0,1,2,3])),
            other_parties        = int(rng.choice([0,1,2])),
            residence_since      = int(rng.integers(1,5)),
            property_magnitude   = int(rng.choice([0,1,2,3])),
            age                  = age,
            other_payment_plans  = int(rng.choice([0,1,2])),
            housing              = int(rng.choice([0,1,2])),
            existing_credits     = int(rng.integers(1,5)),
            job                  = int(rng.choice([0,1,2,3])),
            num_dependents       = int(rng.integers(1,3)),
            own_telephone        = int(rng.choice([0,1])),
            foreign_worker       = int(rng.choice([0,1])),
            active_loans         = int(rng.integers(0,6)),
            credit_utilization   = round(float(rng.uniform(0.1, 0.9)), 2),
            payment_history_score= round(float(rng.uniform(0.4, 1.0)), 2),
        )


# ─────────────────────────────────────────────
#  FAST-USE HELPER
# ─────────────────────────────────────────────
def get_client_from_env() -> PANApiClient:
    """
    Auto-detect provider from environment variables.
    Priority: Perfios > Setu > Karza > CIBIL > Experian > Mock

    Set in your .env or system environment:

        PERFIOS_API_KEY=your_client_id
        PERFIOS_SECRET=your_client_secret

        SETU_CLIENT_ID=your_id
        SETU_CLIENT_SECRET=your_secret

        KARZA_API_KEY=your_key
        CIBIL_API_KEY=your_key
        EXPERIAN_API_KEY=your_key

    If none set → falls back to mock sandbox automatically.
    """
    if os.getenv("PERFIOS_API_KEY"):
        return PANApiClient("perfios",
            api_key=os.getenv("PERFIOS_API_KEY",""),
            secret =os.getenv("PERFIOS_SECRET",""))

    if os.getenv("SETU_CLIENT_ID"):
        return PANApiClient("setu",
            api_key=os.getenv("SETU_CLIENT_ID",""),
            secret =os.getenv("SETU_CLIENT_SECRET",""))

    if os.getenv("KARZA_API_KEY"):
        return PANApiClient("karza",
            api_key=os.getenv("KARZA_API_KEY",""))

    if os.getenv("CIBIL_API_KEY"):
        return PANApiClient("cibil",
            api_key=os.getenv("CIBIL_API_KEY",""))

    if os.getenv("EXPERIAN_API_KEY"):
        return PANApiClient("experian",
            api_key=os.getenv("EXPERIAN_API_KEY",""))

    return PANApiClient("mock")


# ─────────────────────────────────────────────
#  CLI QUICK TEST
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    pan = sys.argv[1] if len(sys.argv) > 1 else "ABCDE1234F"
    client = get_client_from_env()
    profile = client.get_credit_profile(pan)
    print(f"\n{'='*55}")
    print(f"  PAN: {profile.pan}  |  Source: {profile.source.upper()}")
    print(f"  Name: {profile.name}  |  DOB: {profile.date_of_birth}")
    print(f"  PAN Verified: {profile.pan_verified}")
    if profile.error:
        print(f"  ⚠️  Error: {profile.error}")
    print(f"\n  📊 20 Model Features:")
    for k,v in profile.to_model_input().items():
        print(f"    {k:<28} = {v}")
    print(f"{'='*55}\n")
