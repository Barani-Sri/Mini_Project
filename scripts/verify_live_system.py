import sys
import requests

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_URL = "http://127.0.0.1:8000"

def test_endpoint(name, method, path, json_data=None, data=None, params=None, check_fn=None):
    url = f"{BASE_URL}{path}"
    try:
        if method == "GET":
            r = requests.get(url, params=params, timeout=300)
        elif method == "POST":
            r = requests.post(url, json=json_data, data=data, params=params, timeout=300)
        else:
            raise ValueError(f"Unknown method {method}")
        
        if r.status_code != 200:
            print(f"[FAIL] {name}: Failed with status {r.status_code}")
            return False
        
        if check_fn:
            msg = check_fn(r)
            if msg:
                print(f"[FAIL] {name}: {msg}")
                return False
        
        print(f"[PASS] {name}: HTTP {r.status_code} OK")
        return True
    except Exception as e:
        print(f"[FAIL] {name}: Exception: {e}")
        return False

def main():
    print("=" * 65)
    print("LUBDUB MURMUR & HFREF COMPREHENSIVE LIVE SYSTEM VERIFICATION")
    print("=" * 65)

    results = []

    # 1. Frontend HTML Check
    def check_html(r):
        text = r.text
        if "id=\"nav-integrated\"" not in text:
            return "Missing #nav-integrated in HTML"
        if "id=\"view-integrated\"" not in text:
            return "Missing #view-integrated in HTML"
        if "Integrated Result" not in text:
            return "Missing 'Integrated Result' label in HTML"
        return None

    results.append(test_endpoint("1. Frontend index.html & Sidebar Nav", "GET", "/", check_fn=check_html))

    # 2. Existing LubDub System Status
    def check_status(r):
        data = r.json()
        if "model_loaded" not in data:
            return "Missing model_loaded in /api/status"
        return None

    results.append(test_endpoint("2. Existing LubDub Status (/api/status)", "GET", "/api/status", check_fn=check_status))

    # 3. Existing LubDub Demo Analysis
    def check_lubdub_demo(r):
        data = r.json()
        if "patient_id" not in data:
            return "LubDub demo returned no patient_id"
        if "murmur_detection" not in data:
            return "Missing murmur_detection in LubDub demo"
        return None

    results.append(test_endpoint("3. Existing LubDub Demo (/api/demo)", "POST", "/api/demo", check_fn=check_lubdub_demo))

    # 4. HFrEF Status
    def check_hfref_status(r):
        data = r.json()
        if not data.get("is_present"):
            return "Expected is_present=True for HFrEF dataset"
        if data.get("total_subjects") != 500:
            return f"Expected 500 subjects, got {data.get('total_subjects')}"
        return None

    results.append(test_endpoint("4. HFrEF Status (/api/hfref/status)", "GET", "/api/hfref/status", check_fn=check_hfref_status))

    # 5. HFrEF Subjects List
    def check_hfref_subjects(r):
        data = r.json()
        if len(data.get("subjects", [])) < 5:
            return "Expected at least 5 subjects"
        s0 = data["subjects"][0]
        if "subject_id" not in s0 or "lvef" not in s0:
            return "Subject missing subject_id or lvef"
        return None

    results.append(test_endpoint("5. HFrEF Subject List (/api/hfref/subjects)", "GET", "/api/hfref/subjects", params={"limit": 10}, check_fn=check_hfref_subjects))

    # 6. HFrEF Real Subject Analysis (Form data)
    def check_hfref_analyze(r):
        data = r.json()
        if "ecg_analysis" not in data:
            return "Missing ecg_analysis"
        if "locations_analysis" not in data:
            return "Missing locations_analysis"
        if "specialist_recommendation" not in data:
            return "Missing specialist_recommendation"
        return None

    results.append(test_endpoint(
        "6. HFrEF Subject Analysis (/api/hfref/analyze)",
        "POST",
        "/api/hfref/analyze",
        data={"subject_id": "1005263204"},
        check_fn=check_hfref_analyze
    ))

    # 7. Integrated Result Consolidated Endpoint
    def check_integrated_result(r):
        data = r.json()
        if data.get("status") != "success":
            return "Integrated result status not success"
        if "existing_lubdub_result" not in data:
            return "Missing existing_lubdub_result"
        if "new_hfref_ecg_pcg_result" not in data:
            return "Missing new_hfref_ecg_pcg_result"
        if "specialist_recommendation" not in data:
            return "Missing specialist_recommendation"
        return None

    results.append(test_endpoint("7. Consolidated Integrated Result (/api/integrated/result)", "GET", "/api/integrated/result", check_fn=check_integrated_result))

    # 8. Integrated Recommendation Endpoint
    def check_integrated_rec(r):
        data = r.json()
        if "suggested_specialist" not in data:
            return "Missing suggested_specialist"
        if "disclaimer" not in data:
            return "Missing disclaimer"
        return None

    rec_payload = {
        "lubdub_result": {"murmur_detection": {"prediction": "Present", "prediction_class": 1}},
        "hfref_result": {"prediction": "Non-HFrEF", "prediction_class": 0, "estimated_lvef_range": "LVEF > 40%"}
    }
    results.append(test_endpoint("8. Specialist Recommendation (/api/integrated/recommendation)", "POST", "/api/integrated/recommendation", json_data=rec_payload, check_fn=check_integrated_rec))

    # 9. Integrated Report PDF Generation
    def check_pdf(r):
        if r.headers.get("content-type") != "application/pdf":
            return f"Expected application/pdf, got {r.headers.get('content-type')}"
        if r.content[:4] != b"%PDF":
            return "Missing %PDF magic bytes"
        return None

    report_payload = {
        "existing_lubdub_result": {
            "patient_id": "40840",
            "patient_name": "Verified Subject",
            "murmur_detection": {"prediction": "Present", "probability_percent": 52.8},
            "clinical_outcome": {"prediction": "Abnormal", "probability_percent": 51.8}
        },
        "new_hfref_ecg_pcg_result": {
            "subject_id": "1005263204",
            "prediction": "Non-HFrEF",
            "estimated_lvef_range": "LVEF > 40%",
            "hfref_probability_percent": 45.0,
            "ecg_analysis": {"heart_rate_bpm": 53.9}
        },
        "specialist_recommendation": {
            "suggested_specialist": "Cardiologist / Valvular Specialist",
            "priority_level": "Cardiology Auscultation Review",
            "recommendation_text": "Clinical evaluation recommended.",
            "disclaimer": "Research prototype — not a clinical diagnostic system. Findings must be evaluated by a certified medical practitioner."
        }
    }
    results.append(test_endpoint("9. Integrated PDF Export (/api/integrated/report)", "POST", "/api/integrated/report", json_data=report_payload, check_fn=check_pdf))

    print("=" * 65)
    passed = sum(1 for r in results if r)
    total = len(results)
    print(f"VERIFICATION SUMMARY: {passed}/{total} CHECKS PASSED")
    print("=" * 65)

    if passed != total:
        sys.exit(1)

if __name__ == "__main__":
    main()
