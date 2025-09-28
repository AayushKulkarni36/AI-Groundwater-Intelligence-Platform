from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import os
import sys
import json
import re
from typing import Optional, Dict, Any

app = Flask(__name__)
CORS(app)

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_PATH = os.path.join(ROOT_DIR, 'data', 'groundwater_data.csv')
# Ensure project root is on sys.path for `ml` imports when running as a script
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
from ml.forecast import forecast_stage
from ml.simulation import simulate_scenario

# --- OpenAI helpers (optional) ---
_OPENAI_CLIENT = None

def _get_openai_chat_response(model: str, messages: list) -> Optional[str]:
    global _OPENAI_CLIENT
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        return None
    # Try new SDK first
    try:
        if _OPENAI_CLIENT is None:
            from openai import OpenAI  # type: ignore
            _OPENAI_CLIENT = OpenAI()
        resp = _OPENAI_CLIENT.chat.completions.create(model=model, messages=messages)
        return resp.choices[0].message.content
    except Exception:
        # Fallback to legacy SDK
        try:
            import openai  # type: ignore
            openai.api_key = api_key
            resp = openai.ChatCompletion.create(model=model, messages=messages)
            return resp.choices[0].message["content"]
        except Exception:
            return None


def parse_query_llm(user_query: str) -> Dict[str, Any]:
    """Use LLM (if available) to extract structured fields from a user query.
    Returns dict with keys: district (str|None), year (int|None), intent ('query_status'|'forecast'|'simulate'),
    years_ahead (int|None), extraction_change (float|None), recharge_change (float|None).
    Falls back to regex if LLM unavailable.
    """
    prompt = (
        "You are a groundwater policy assistant. Extract STRICT JSON with keys: "
        "district (string|null), year (integer|null), intent ('query_status'|'forecast'|'simulate'), "
        "years_ahead (integer|null), extraction_change (float|null), recharge_change (float|null).\n"
        "If user mentions what-if scenarios, set intent='simulate' and parse percentage deltas (negative for decrease).\n"
        "Examples:\n"
        "Q: 'How is groundwater in Pune 2022?' => {\"district\": \"Pune\", \"year\": 2022, \"intent\": \"query_status\", \"years_ahead\": null, \"extraction_change\": null, \"recharge_change\": null}\n"
        "Q: 'Forecast Pune for 3 years' => {\"district\": \"Pune\", \"year\": null, \"intent\": \"forecast\", \"years_ahead\": 3, \"extraction_change\": null, \"recharge_change\": null}\n"
        "Q: 'What if extraction decreases by 15% in Pune for the next 3 years?' => {\"district\": \"Pune\", \"intent\": \"simulate\", \"years_ahead\": 3, \"extraction_change\": -15, \"recharge_change\": null, \"year\": null}\n"
        "Q: 'If recharge increases by 10% in Nashik' => {\"district\": \"Nashik\", \"intent\": \"simulate\", \"recharge_change\": 10, \"years_ahead\": null, \"extraction_change\": null, \"year\": null}\n"
        f"Q: '{user_query}' => "
    )
    content = _get_openai_chat_response(
        model=os.environ.get('OPENAI_MODEL', 'gpt-4o-mini'),
        messages=[{"role": "system", "content": "You are a strict JSON extractor."}, {"role": "user", "content": prompt}],
    )
    result = {"district": None, "year": None, "intent": "query_status", "years_ahead": None}
    if content:
        try:
            parsed = json.loads(content)
            result.update({
                "district": parsed.get("district"),
                "year": parsed.get("year"),
                "intent": parsed.get("intent") or "query_status"
            })
            # Normalize year
            if result["year"] is not None:
                result["year"] = int(result["year"])
        except Exception:
            pass
    # Fallback regex logic
    ql = user_query.strip()
    # 1) Try forecast pattern: "forecast <district> for <n> years"
    mf = re.search(r"(?i)forecast\s+([A-Za-z\-\s]+)(?:\s+for\s+(\d+)\s+years?)?$", ql)
    if mf:
        result["district"] = result["district"] or (mf.group(1).strip() if mf.group(1) else None)
        if mf.group(2):
            result["years_ahead"] = int(mf.group(2))
        result["intent"] = "forecast"
        return result

    # 1b) What-if pattern: extraction/recharge +/- N%
    mw = re.search(r"(?i)(?:what\s*if\s*)?(?:.*?)(extraction|recharge)\s+(?:increases?|decreases?|reduced|reduction|increase|decrease)\s+by\s+(\d+)%\s+in\s+([A-Za-z\-\s]+)(?:.*?(\d+)\s+years?)?", ql)
    if mw:
        kind = mw.group(1).lower()
        delta = float(mw.group(2))
        dname = mw.group(3).strip()
        yrs = mw.group(4)
        if 'decrease' in mw.group(0).lower() or 'reduced' in mw.group(0).lower():
            delta = -delta
        result['district'] = result['district'] or dname
        result['intent'] = 'simulate'
        if kind == 'extraction':
            result['extraction_change'] = delta
        else:
            result['recharge_change'] = delta
        if yrs:
            result['years_ahead'] = int(yrs)
        return result
    if result["district"] is None or (result["year"] is None and "forecast" not in str(result["intent"])):
        # 2) Status pattern: 'status of <district> in <year>' or '<district> <year>' (allow trailing punctuation)
        m = re.search(r"(?i)(?:status\s+of\s+)?([A-Za-z\-\s]+?)(?:\s+in\s+(\d{4}))?\s*[\?\.!]*$", ql)
        if m:
            cand = (m.group(1).strip() if m.group(1) else None)
            result["district"] = result["district"] or cand
            if m.group(2):
                result["year"] = int(m.group(2))

    # 3) Heuristic fallback using known districts and year in text
    if result["district"] is None:
        ql_low = ql.lower()
        try:
            for dname in df['District'].unique():
                if str(dname).lower() in ql_low:
                    result["district"] = str(dname)
                    break
        except Exception:
            pass
    if result["year"] is None:
        years = re.findall(r"\b(19|20)\d{2}\b", ql)
        if years:
            # re.findall captured the century prefix group; extract full 4-digit via another findall
            years_full = re.findall(r"\b\d{4}\b", ql)
            if years_full:
                result["year"] = int(years_full[-1])

    # 4) Infer intent if not set
    if not result.get("intent"):
        result["intent"] = "forecast" if re.search(r"(?i)\b(forecast|predict|projection)\b", ql) else "query_status"

    return result


def generate_answer_llm(context: Dict[str, Any]) -> Optional[str]:
    """Optionally generate a conversational answer from context using LLM.
    Returns None if LLM not available.
    """
    messages = [
        {"role": "system", "content": "You are an assistant summarizing groundwater status for users in simple terms."},
        {"role": "user", "content": json.dumps(context)}
    ]
    return _get_openai_chat_response(model=os.environ.get('OPENAI_MODEL', 'gpt-4o-mini'), messages=messages)

df = pd.read_csv(DATA_PATH)
# Normalize column names to expected keys
rename_map = {}
if 'Recharge (MCM)' in df.columns:
    rename_map['Recharge (MCM)'] = 'Recharge'
if 'Extraction (MCM)' in df.columns:
    rename_map['Extraction (MCM)'] = 'Extraction'
if 'Stage (%)' in df.columns:
    rename_map['Stage (%)'] = 'Stage'
if rename_map:
    df = df.rename(columns=rename_map)
# Types
df['Year'] = df['Year'].astype(int)

@app.get('/health')
def health():
    return jsonify({"status": "ok"})

@app.get('/query')
def query():
    # Accept raw natural language too
    q = request.args.get('q')
    state = request.args.get('state')
    district = request.args.get('district')
    year = request.args.get('year', type=int)

    intent = 'query_status'
    ya_parsed = None
    ex_parsed = None
    rc_parsed = None
    if q:
        parsed = parse_query_llm(q)
        intent = parsed.get('intent') or intent
        district = district or parsed.get('district')
        if year is None:
            year = parsed.get('year')
        ya_parsed = parsed.get('years_ahead')
        ex_parsed = parsed.get('extraction_change')
        rc_parsed = parsed.get('recharge_change')

    if not district:
        return jsonify({"error": "district is required"}), 400

    # Case-insensitive filtering
    subset = df[df['District'].str.lower() == district.lower()]
    if state:
        subset = subset[subset['State'].str.lower() == state.lower()]

    subset = subset.sort_values('Year')

    match = None
    if year is not None:
        matched_rows = subset[subset['Year'] == year]
        if not matched_rows.empty:
            match = matched_rows.iloc[0].to_dict()

    history = subset.to_dict(orient='records')

    # Optional forecast branch if requested
    predictions = None
    simulation = None
    if intent == 'forecast':
        try:
            ya = request.args.get('years_ahead', type=int)
            if ya is None:
                ya = ya_parsed or 3
            predictions = forecast_stage(district, years_ahead=int(ya)).to_dict(orient='records')
        except Exception:
            predictions = []
    elif intent == 'simulate':
        try:
            ex = request.args.get('extraction_change', type=float)
            rc = request.args.get('recharge_change', type=float)
            ya = request.args.get('years_ahead', type=int)
            if ex is None:
                ex = ex_parsed if ex_parsed is not None else 0.0
            if rc is None:
                rc = rc_parsed if rc_parsed is not None else 0.0
            if ya is None:
                ya = ya_parsed or 5
            simulation = simulate_scenario(district, extraction_change=float(ex), recharge_change=float(rc), years_ahead=int(ya)).to_dict(orient='records')
        except Exception:
            simulation = []

    # Build answer context and optionally call LLM to phrase a reply
    context = {
        "query": q,
        "intent": intent,
        "district": district,
        "year": year,
        "match": match,
        "history": history,
        "predictions": predictions,
    }
    answer = generate_answer_llm(context) or _default_answer(context)

    return jsonify({
        "intent": intent,
        "district": district,
        "year": year,
        "match": match,
        "history": history,
        "predictions": predictions,
        "simulation": simulation,
        "answer": answer,
    })


def _default_answer(ctx: Dict[str, Any]) -> str:
    # Simple templated fallback if LLM not configured
    d = ctx.get('district')
    y = ctx.get('year')
    m = ctx.get('match')
    if ctx.get('intent') == 'forecast' and ctx.get('predictions'):
        last = ctx['predictions'][-1]
        return f"Forecast for {d}: by {last['Year']} expected Stage {last['PredictedStage']}% ({last['PredictedCategory']})."
    if ctx.get('intent') == 'simulate' and ctx.get('simulation'):
        last = ctx['simulation'][-1]
        return f"What-if for {d}: by {last['Year']} projected Stage {last['PredictedStage']}% ({last['PredictedCategory']})."
    if m:
        return f"In {d} {m['Year']}, category: {m['Category']} at {m['Stage']}% stage."
    return f"Showing history for {d}. Select a year for details."

@app.get('/forecast')
def forecast():
    district = request.args.get('district')
    years_ahead = request.args.get('years_ahead', default=3, type=int)
    if not district:
        return jsonify({"error": "district is required"}), 400
    try:
        pred_df = forecast_stage(district, years_ahead=years_ahead)
        return jsonify({
            "district": district,
            "years_ahead": years_ahead,
            "predictions": pred_df.to_dict(orient='records')
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.get('/simulate')
def simulate():
    district = request.args.get('district')
    ex = request.args.get('extraction_change', default=0.0, type=float)
    rc = request.args.get('recharge_change', default=0.0, type=float)
    ya = request.args.get('years_ahead', default=5, type=int)
    if not district:
        return jsonify({"error": "district is required"}), 400
    try:
        sim = simulate_scenario(district, extraction_change=ex, recharge_change=rc, years_ahead=ya)
        return jsonify({
            "district": district,
            "extraction_change": ex,
            "recharge_change": rc,
            "years_ahead": ya,
            "results": sim.to_dict(orient='records')
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
