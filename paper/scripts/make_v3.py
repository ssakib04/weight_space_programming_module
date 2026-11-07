"""v3 revision: selected title, dated author attestation for Cursor session hygiene, CIFAR moved to bounded exploratory appendix.
Edits only manuscript text/layout. No raw data, scores, or experiments change."""
from pathlib import Path
from bs4 import BeautifulSoup
import re
P = Path(__file__).resolve().parents[1]
f = P / 'WSP_manuscript_v3.html'
h = f.read_text()
TITLE = 'How Reliably Can Language Models Construct Neural Network Weights? Evidence from Modular Arithmetic Tasks'
def rep(old, new, count=1):
    global h
    assert old in h, old[:80]
    h = h.replace(old, new, count)
rep('<title>Can Language Models Construct Neural Network Weights?</title>', f'<title>{TITLE}</title><meta name="author" content="Kazi Sadman Sakib"/>')
rep('<h1>Can Language Models Construct Neural Network Weights?<br/>\nAn Exploratory Study of Modular Arithmetic Tasks</h1>',
    '<h1>How Reliably Can Language Models Construct Neural Network Weights?<br/>\nEvidence from Modular Arithmetic Tasks</h1>')
banner_old = h[h.index('<div class="banner">'):h.index('</div>', h.index('<div class="banner">')) + 6]
h = h.replace(banner_old, '<div class="banner"><b>REVIEW MANUSCRIPT v3 — PRE-PUBLICATION</b> (built 7 Nov 2026, Asia/Dhaka). Not posted, not submitted, not peer reviewed. '
    'Revises the audited v2 build with the author-selected title, a dated author attestation for Cursor session conditions, and CIFAR-10 moved to an exploratory appendix. '
    'No new training, evaluations, or model calls. Earlier drafts and the audited v2 snapshot are preserved unchanged.</div>')
rep('Across separately labelled channels we find that closed chat models often succeed on these tasks.',
    'Across separately labelled channels we measure how often closed chat models succeed on these tasks, at small, exploratory sample sizes.')
rep('On a Cursor chat channel (user-reported labels; no-hint prompts; <i>n</i>=5 per cell)',
    'On a Cursor chat channel (no-hint prompts; <i>n</i>=5 per model×task cell; model labels user-reported; session conditions author-attested)')
rep(' The Cursor count is mathematical construction success after documented extraction, including one formatting exception, not a strict raw-response compliance rate.</p>',
    ' The Cursor count is mathematical construction success after documented extraction, including one formatting exception, not a strict raw-response compliance rate. '
    'Reliability estimates are exploratory: with <i>n</i>=5 per cell, the per-cell 95% intervals are wide, and channels are not a controlled head-to-head ranking.</p>')
rep('Channels differ in interface and confirmation of chat hygiene; we report them separately and do not pool them into a leaderboard.',
    'Channels differ in interface, sampling control, and how session conditions are documented; we report them separately and do not pool them into a leaderboard.')
rep('no-hint; 5×2×3=30; fresh Ask not newly confirmed; protocol package frozen post-pilot for that channel only.',
    'no-hint; 5×2×3=30; protocol package frozen post-pilot for that channel only. <b>Session conditions (author attestation, recorded 7 Nov 2026, 13:20 Asia/Dhaka):</b> '
    'the author states that every Cursor query used Ask mode, the frozen prompt pasted unchanged, an empty workspace, and a separate chat per query. '
    'This statement was given after collection and is recorded as an attestation; it was not independently verified from Cursor logs, and contemporaneous trial metadata are left unchanged. '
    'Model labels are those shown in the Cursor model picker as reported by the author; provider-side model identities, versions, and sampling settings are not observable in this channel.')
rep('Provider identities and fresh-chat hygiene are not independently verified for Cursor.</p>',
    'Provider identities are user-reported and session conditions are author-attested (above); neither is independently verified.</p>')
# Move CIFAR to a bounded exploratory appendix.
cs = h.index('<h3>4.5 CIFAR-10 (exploratory)</h3>'); ce = h.index('<h2>5. What successful constructions do</h2>')
cifar = h[cs:ce]; h = h[:cs] + '<p>CIFAR-10 results are outside the modular-arithmetic scope and are reported separately in Appendix B as exploratory endpoints only.</p>\n' + h[ce:]
cifar = cifar.replace('<h3>4.5 CIFAR-10 (exploratory)</h3>', '<h2>Appendix B. CIFAR-10 exploratory endpoints</h2>\n<p>We include these logged endpoints for completeness. '
    'By the author\'s scoping decision, this paper makes <b>no</b> claim that LLM-written filters provide a feature advantage, and no matched random-filter controls were run; '
    'such controls would be required for any feature-quality claim and are left to future work. Pure LLM-written CNN weights were near chance; '
    'the B+ridge rows use 10,000 labelled training images for a closed-form readout and are therefore data-assisted diagnostics, not weight construction without data.</p>\n')
cifar = cifar.replace('No matched random-filter control.</figcaption>', 'Exploratory only; no random-filter control and no feature-advantage claim.</figcaption>')
rep('<li>Cursor labels/fresh-Ask unconfirmed.</li>', '<li>Cursor model labels user-reported; session conditions author-attested after collection, not independently verified.</li>')
rep('<li>CIFAR without random-filter controls.</li>', '<li>CIFAR-10 is an exploratory appendix only; without random-filter controls we make no feature-advantage claim.</li>')
rep('<li>Small <i>n</i> (Cursor 5, local 3).</li>', '<li>Small <i>n</i> (Cursor 5 per cell, API 10–15 per cell, local 3 per cell): reliability rates are exploratory estimates with wide intervals.</li>')
rep('Sparks of Weight-Space Programming. MIT CSAIL blog, 2 Nov 2026. https://rikiyat.github.io/blog/weight-space-programming/',
    'Sparks of Weight-Space Programming: LLMs can natively read &amp; write neural network weights. Blog post, November 2026. https://rikiyat.github.io/blog/weight-space-programming/')
rep('coin Weight-Space Programming in a MIT CSAIL blog post', 'coin Weight-Space Programming in a blog post (MIT CSAIL authors)') if 'coin Weight-Space Programming in a MIT CSAIL blog post' in h else None
# Appendix A paths -> release-candidate relative layout.
a0 = h.index('<h2>Appendix A. Artifact paths</h2>'); a1 = h.index('</ul>', a0) + 5
h = h[:a0] + ('<h2>Appendix A. Artifacts</h2>\n<ul>\n<li>Release candidate (local, not yet published): <span class="mono">github_release_candidate/</span> — frozen prompts, original responses, saved weights for all 30 Cursor trials, API and open-model scores, sanitized notebooks, offline evaluator.</li>\n'
    '<li>Figure sources and hashes: <span class="mono">SOURCE_TO_FIGURE_MANIFEST.md</span>.</li>\n<li>Cursor session attestation addendum: <span class="mono">cursor_trials/ATTESTATION_ADDENDUM_2026-11-07.json</span>.</li>\n</ul>\n') + cifar + h[a1:]
s = BeautifulSoup(h, 'html.parser')
for foot in s.select('.foot'): foot.decompose()
# Renumber figures in reading order; keep stable filenames.
old2new = {}
for n, fig in enumerate(s.find_all('figure'), 1):
    cap = fig.find('figcaption'); t = cap.get_text(); old = int(re.match(r'Figure (\d+)\.', t).group(1))
    old2new[old] = n; cap.string = re.sub(r'^Figure \d+\.', f'Figure {n}.', t)
for node in list(s.find_all(string=re.compile(r'Fig\.\s\d+'))):
    node.replace_with(re.sub(r'Fig\.(\s)(\d+)', lambda m: f'Fig.{m.group(1)}{old2new[int(m.group(2))]}', str(node)))
f.write_text(str(s))
print('figures', len(s.find_all('figure')), 'tables', len(s.find_all('table')), 'map', old2new)
