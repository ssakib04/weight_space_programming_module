# gemini4argonmedium_modmul_none_r4

Model: Gemini 4 Argon Medium (t89u batch). Ask confirmed for r0 (t82); this fresh chat Ask **not newly confirmed**.
g=3 hardcoded as-is. Construction primary. No numerical repair.

- valid/ok: True
- success (>=0.99): False
- harness all_acc: 0.24920280277729034
- exact torch: {'correct': 3126, 'total': 12544, 'wrong': 9418, 'accuracy': 0.24920280612244897, 'param_dtype': 'torch.float32'}
- np64: {'dtype': "<class 'numpy.float64'>", 'correct': 4904, 'total': 12544, 'wrong': 7640, 'accuracy': 0.3909438775510204, 'min_margin_raw': -56.00000013461476, 'mean_margin_raw': -1.278467105685763, 'finite_logits': True}
- np32: {'dtype': "<class 'numpy.float32'>", 'correct': 3126, 'total': 12544, 'wrong': 9418, 'accuracy': 0.24920280612244897, 'min_margin_raw': -56.000022888183594, 'mean_margin_raw': -1.2784734964370728, 'finite_logits': True}
- source dtype intent: float32
- min_margin_raw harness: -56.000022888183594
- connected: {'W1_nonzero_rows': 222, 'W2_nonzero_cols': 222, 'n_connected': 222}
- notes: Fourth construction may be mathematically flawed; scored as-is without repair
- error: None None
