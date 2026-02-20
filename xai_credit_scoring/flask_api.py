"""
flask_api.py — Flask REST API for FinTrust AI Credit Scoring
=============================================================
Run alongside Streamlit:
    python flask_api.py          → starts on http://localhost:5000
    streamlit run app.py         → starts on http://localhost:8501

Endpoints:
    GET  /api/health             → System status
    POST /api/predict            → Predict from 20 features
    POST /api/pan-check          → Full PAN-based credit check
    GET  /api/features           → List of required model features
    POST /api/simulate           → What-if simulation (before/after)
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import re
import pandas as pd
import shap
import time

from pan_api_client import PANApiClient, get_client_from_env

# ─────────────────────────────────────────────
#  APP INIT
# ─────────────────────────────────────────────
app = Flask(__name__)
CORS(app)  # Allow cross-origin requests (for frontend integration)

# Load model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# SHAP explainer (cached)
explainer = shap.TreeExplainer(model)

# API client (auto-detect from env, or mock)
pan_client = get_client_from_env()

# Feature names (order matters — must match training)
FEATURE_NAMES = [
    'checking_status', 'duration', 'credit_history', 'purpose',
    'credit_amount', 'savings_status', 'employment', 'installment_commitment',
    'personal_status', 'other_parties', 'residence_since', 'property_magnitude',
    'age', 'other_payment_plans', 'housing', 'existing_credits',
    'job', 'num_dependents', 'own_telephone', 'foreign_worker'
]


def cibil_score(prob):
    """Convert default probability → CIBIL scale (300-900)"""
    return int(900 - prob * 600)


def get_decision(score):
    """Get loan decision based on CIBIL score"""
    if score >= 750:
        return {"decision": "AUTO-APPROVED", "color": "green", "reason": "Excellent creditworthiness"}
    elif score >= 650:
        return {"decision": "MANUAL REVIEW", "color": "yellow", "reason": "Good profile, needs underwriter check"}
    elif score >= 550:
        return {"decision": "MANUAL REVIEW", "color": "orange", "reason": "Fair profile, higher scrutiny needed"}
    else:
        return {"decision": "REJECTED", "color": "red", "reason": "High default risk"}


# ─────────────────────────────────────────────
#  ENDPOINT: Home (Root)
# ─────────────────────────────────────────────
@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "name": "FinTrust AI — Credit Scoring API",
        "version": "2.0",
        "status": "online",
        "endpoints": {
            "GET  /": "This page",
            "GET  /api/health": "System status",
            "GET  /api/features": "List of required model features",
            "POST /api/predict": "Predict credit score from 20 features",
            "POST /api/pan-check": "Full PAN-based credit check",
            "POST /api/simulate": "What-if simulation (before/after)"
        }
    })


# ─────────────────────────────────────────────
#  ENDPOINT: Health Check
# ─────────────────────────────────────────────
@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        "status": "online",
        "model": type(model).__name__,
        "features_count": len(FEATURE_NAMES),
        "bureau_provider": pan_client.provider,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "version": "2.0"
    })


# ─────────────────────────────────────────────
#  ENDPOINT: List Features
# ─────────────────────────────────────────────
@app.route('/api/features', methods=['GET'])
def list_features():
    feature_info = {
        'checking_status':      {"range": "0-3", "description": "Checking account status (0=no account, 3=healthy)"},
        'duration':             {"range": "6-72", "description": "Loan duration in months"},
        'credit_history':       {"range": "0-4", "description": "Credit history quality (0=bad, 4=perfect)"},
        'purpose':              {"range": "0-9", "description": "Loan purpose (0=car new, 7=vacation, 9=business)"},
        'credit_amount':        {"range": "500-15000", "description": "Loan amount requested"},
        'savings_status':       {"range": "0-4", "description": "Savings account balance (0=none, 4=rich)"},
        'employment':           {"range": "0-4", "description": "Employment duration (0=unemployed, 4=7+ years)"},
        'installment_commitment': {"range": "1-4", "description": "Installment rate as % of income"},
        'personal_status':      {"range": "0-3", "description": "Personal status/gender"},
        'other_parties':        {"range": "0-2", "description": "Other parties (0=none, 2=guarantor)"},
        'residence_since':      {"range": "1-4", "description": "Years at current residence"},
        'property_magnitude':   {"range": "0-3", "description": "Property/collateral (0=none, 3=real estate)"},
        'age':                  {"range": "18-80", "description": "Applicant age"},
        'other_payment_plans':  {"range": "0-2", "description": "Other payment plans (0=none)"},
        'housing':              {"range": "0-2", "description": "Housing (0=free, 1=rent, 2=own)"},
        'existing_credits':     {"range": "1-4", "description": "Existing credits at this bank"},
        'job':                  {"range": "0-3", "description": "Job skill level (0=unskilled, 3=highly skilled)"},
        'num_dependents':       {"range": "1-2", "description": "Number of dependents"},
        'own_telephone':        {"range": "0-1", "description": "Has registered telephone"},
        'foreign_worker':       {"range": "0-1", "description": "Is foreign worker"},
    }
    return jsonify({"features": feature_info, "total": len(feature_info)})


# ─────────────────────────────────────────────
#  ENDPOINT: Predict from Features
# ─────────────────────────────────────────────
@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Predict credit score from 20 features.
    
    Request body (JSON):
    {
        "checking_status": 1,
        "duration": 24,
        "credit_history": 3,
        ...all 20 features...
    }
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "Request body must be JSON with 20 features"}), 400

    # Validate all features present
    missing = [f for f in FEATURE_NAMES if f not in data]
    if missing:
        return jsonify({
            "error": f"Missing features: {missing}",
            "required": FEATURE_NAMES
        }), 400

    try:
        # Build input DataFrame
        features = {f: [data[f]] for f in FEATURE_NAMES}
        idf = pd.DataFrame(features)

        # Predict
        prob = float(model.predict_proba(idf)[0][1])
        score = cibil_score(prob)
        decision = get_decision(score)

        # SHAP explanation
        sv = explainer(idf)
        shap_values = {}
        for i, feat in enumerate(FEATURE_NAMES):
            shap_values[feat] = {
                "value": float(sv.values[0][i]),
                "impact": "risk_increasing" if sv.values[0][i] > 0 else "risk_decreasing"
            }

        # Sort by absolute impact
        top_factors = sorted(shap_values.items(), key=lambda x: abs(x[1]['value']), reverse=True)[:5]

        return jsonify({
            "success": True,
            "default_probability": round(prob, 4),
            "cibil_score": score,
            "grade": decision["decision"],
            "reason": decision["reason"],
            "risk_level": "LOW" if score >= 750 else ("MEDIUM" if score >= 600 else "HIGH"),
            "shap_explanation": shap_values,
            "top_5_factors": [
                {"feature": f, "impact": v['value'], "direction": v['impact']}
                for f, v in top_factors
            ],
            "input_features": data
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────
#  ENDPOINT: PAN Card Check
# ─────────────────────────────────────────────
@app.route('/api/pan-check', methods=['POST'])
def pan_check():
    """
    Full PAN-based credit check.
    
    Request body (JSON):
    {
        "pan": "ABCDE1234F",
        "name": "Raj Kumar",       (optional)
        "age": 30                   (optional)
    }
    """
    data = request.get_json()
    if not data or 'pan' not in data:
        return jsonify({"error": "Request body must include 'pan' field"}), 400

    pan = data['pan'].upper().strip()

    # Validate PAN format

    if not re.match(r'^[A-Z]{5}[0-9]{4}[A-Z]{1}$', pan):
        return jsonify({"error": f"Invalid PAN format: {pan}. Expected: ABCDE1234F"}), 400

    try:
        # Fetch profile from bureau
        profile = pan_client.get_credit_profile(pan)
        feats = profile.to_model_input()

        # Override age if provided
        if 'age' in data:
            feats['age'] = int(data['age'])

        # Predict
        idf = pd.DataFrame([feats])
        prob = float(model.predict_proba(idf)[0][1])
        score = cibil_score(prob)
        decision = get_decision(score)

        # SHAP top factors
        sv = explainer(idf)
        top_factors = []
        for i, feat in enumerate(FEATURE_NAMES):
            top_factors.append((feat, float(sv.values[0][i])))
        top_factors.sort(key=lambda x: abs(x[1]), reverse=True)

        return jsonify({
            "success": True,
            "pan": pan,
            "name": profile.name,
            "date_of_birth": profile.date_of_birth,
            "pan_verified": profile.pan_verified,
            "bureau_source": profile.source,
            "default_probability": round(prob, 4),
            "cibil_score": score,
            "grade": decision["decision"],
            "reason": decision["reason"],
            "risk_level": "LOW" if score >= 750 else ("MEDIUM" if score >= 600 else "HIGH"),
            "monthly_income": profile.monthly_income,
            "foir": profile.foir,
            "risk_band": profile.perfios_risk_band,
            "top_5_factors": [
                {"feature": f, "shap_value": round(v, 4),
                 "direction": "risk_increasing" if v > 0 else "risk_decreasing"}
                for f, v in top_factors[:5]
            ],
            "all_features": feats,
            "bureau_error": profile.error
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────
#  ENDPOINT: What-If Simulation
# ─────────────────────────────────────────────
@app.route('/api/simulate', methods=['POST'])
def simulate():
    """
    Compare two scenarios — baseline vs modified.
    
    Request body (JSON):
    {
        "baseline": { ...20 features... },
        "modified": { ...20 features with changes... }
    }
    """
    data = request.get_json()
    if not data or 'baseline' not in data or 'modified' not in data:
        return jsonify({"error": "Must provide 'baseline' and 'modified' feature sets"}), 400

    try:
        base_feats = data['baseline']
        mod_feats  = data['modified']

        # Predict both
        base_idf = pd.DataFrame([{f: base_feats[f] for f in FEATURE_NAMES}])
        mod_idf  = pd.DataFrame([{f: mod_feats[f] for f in FEATURE_NAMES}])

        base_prob = float(model.predict_proba(base_idf)[0][1])
        mod_prob  = float(model.predict_proba(mod_idf)[0][1])

        base_score = cibil_score(base_prob)
        mod_score  = cibil_score(mod_prob)

        # What changed
        changes = {}
        for f in FEATURE_NAMES:
            if base_feats.get(f) != mod_feats.get(f):
                changes[f] = {"from": base_feats.get(f), "to": mod_feats.get(f)}

        return jsonify({
            "success": True,
            "baseline": {
                "score": base_score,
                "probability": round(base_prob, 4),
                "decision": get_decision(base_score)["decision"]
            },
            "modified": {
                "score": mod_score,
                "probability": round(mod_prob, 4),
                "decision": get_decision(mod_score)["decision"]
            },
            "delta": {
                "score_change": mod_score - base_score,
                "probability_change": round((mod_prob - base_prob) * 100, 2),
                "improved": mod_score > base_score
            },
            "changes_made": changes
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────
#  RUN SERVER
# ─────────────────────────────────────────────
if __name__ == '__main__':
    print("\n" + "="*55)
    print("  🏦 FinTrust AI — Flask REST API")
    print("  📍 Running on: http://localhost:5000")
    print("  📊 Model: " + type(model).__name__)
    print("  🔗 Bureau: " + pan_client.provider.upper())
    print("="*55)
    print("\n  Endpoints:")
    print("    GET  /api/health      → System status")
    print("    GET  /api/features    → Feature list")
    print("    POST /api/predict     → Score from features")
    print("    POST /api/pan-check   → PAN credit check")
    print("    POST /api/simulate    → What-if comparison")
    print("="*55 + "\n")

    app.run(debug=True, port=5000)
