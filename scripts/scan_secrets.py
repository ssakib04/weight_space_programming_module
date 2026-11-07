"""Scan the release tree for credential-like patterns and private details. Prints counts only, never matched values."""
import re, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
PATTERNS = {
    'anthropic_key': r'sk-ant-[A-Za-z0-9_\-]{20,}', 'openai_style_key': r'\bsk-[A-Za-z0-9]{32,}',
    'huggingface_token': r'\bhf_[A-Za-z0-9]{30,}', 'github_token': r'\bgh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}',
    'aws_access_key': r'\bAKIA[0-9A-Z]{16}\b', 'private_key_block': r'-----BEGIN [A-Z ]*PRIVATE KEY-----',
    'jwt': r'\beyJ[A-Za-z0-9_\-]{20,}\.[A-Za-z0-9_\-]{20,}\.', 'slack_token': r'\bxox[abpr]-[A-Za-z0-9\-]{10,}',
    'kaggle_key_value': r'"key"\s*:\s*"[a-f0-9]{32}"', 'secret_assignment': r'(?i)(api_key|token|password|secret)\s*=\s*["\'][A-Za-z0-9_\-]{16,}["\']',
    'email_address': r'[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+\.(com|org|net|io|edu|ai)\b',
    'private_path': r'/workspace/|/home/[a-z]', 'kaggle_account_url': r'kaggle\.com/code/[A-Za-z0-9_\-]+',
}
def scan(root=ROOT):
    hits = {}
    for p in root.rglob('*'):
        if not p.is_file() or p.suffix in {'.npz', '.png', '.pdf', '.zip'} or p.resolve() == Path(__file__).resolve() or '__pycache__' in p.parts or 'verification_output' in p.parts: continue
        t = p.read_text(errors='ignore')
        for name, rx in PATTERNS.items():
            n = len(re.findall(rx, t))
            if n: hits.setdefault(name, []).append((str(p.relative_to(root)), n))
    return hits
if __name__ == '__main__':
    h = scan(Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT)
    for k, v in h.items(): print(k, sum(n for _, n in v), 'in', [f for f, _ in v][:8])
    print('SECRET SCAN:', 'CLEAN' if not h else 'REVIEW NEEDED'); sys.exit(1 if h else 0)
