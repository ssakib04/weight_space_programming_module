# gemini4argonmedium_modmul_none_r2

Model: Gemini 4 Argon Medium (t89u batch). Ask confirmed for r0 (t82); this fresh chat Ask **not newly confirmed**.
g=3 hardcoded as-is. Construction primary. No numerical repair.

- valid/ok: True
- success (>=0.99): False
- harness all_acc: 0.009247449226677418
- exact torch: {'correct': 116, 'total': 12544, 'wrong': 12428, 'accuracy': 0.009247448979591837, 'param_dtype': 'torch.float32'}
- np64: {'dtype': "<class 'numpy.float64'>", 'correct': 116, 'total': 12544, 'wrong': 12428, 'accuracy': 0.009247448979591837, 'min_margin_raw': -138.8784888221748, 'mean_margin_raw': -55.29714583557658, 'finite_logits': True}
- np32: {'dtype': "<class 'numpy.float32'>", 'correct': 116, 'total': 12544, 'wrong': 12428, 'accuracy': 0.009247448979591837, 'min_margin_raw': -138.8784942626953, 'mean_margin_raw': -55.29714584350586, 'finite_logits': True}
- source dtype intent: float64
- min_margin_raw harness: -138.8784942626953
- connected: {'W1_nonzero_rows': 225, 'W2_nonzero_cols': 225, 'n_connected': 225}
- notes: None
- error: None None
