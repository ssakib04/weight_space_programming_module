import os, sys, tempfile
HERE = os.path.dirname(os.path.abspath(__file__)); SRC = os.path.join(HERE, "..", "v3_src")
def load(files=("01_config.py", "10_tasks.py", "20_harness.py", "30_llm.py", "50_B.py", "60_cifar.py"), overrides=None, env=None, out=None):
    out = out or tempfile.mkdtemp(prefix="wsp_t_")
    os.environ.update({"WSP_OUT": out, "WSP_CACHE_DIR": os.path.join(out, "cache"), "WSP_V2_LLM_LOG": "", "WSP_FAKE_CIFAR": "1",
                       "WSP_QUICK": "1", **(env or {})})
    ns = {"__name__": "__main__"}
    for f in files:
        exec(compile(open(os.path.join(SRC, f)).read(), f, "exec"), ns)
        if f == "01_config.py":
            ns["RUN_A"] = False
            ns.update(overrides or {})
    return ns
