# FP-RNN

This code accompanies the paper:\
\[[NeurIPS'2025](https://neurips.cc/)\] \[[Fixed-Point RNNs: Interpolating from Diagonal to Dense](https://arxiv.org/abs/2503.10799)\]\
Sajad Movahedi, Felix Sarnthein, Nicola Muca Cirone, Antonio Orvieto

![](fp-overview_v4.png)

---

## Setup

Use pip with the provided requirements:
```bash
python -m venv .venv && source .venv/bin/activate
# CPU
pip install -r requirements.txt
# GPU (specify the torch CUDA index)
pip install --extra-index-url https://download.pytorch.org/whl/cu121 -r requirements.txt
# Optional CUDA conv kernels
pip install causal-conv1d
```

---

## Running

- Copying: `bash run_copy_exp.sh`
- Word problems: `bash run_word_problem_exp.sh`
- Arithmetic: `bash run_arithmetic_exp.sh`

Adjust script flags as needed (see the paper for details).

---

## Credits

- Language modeling pipeline adapted from https://github.com/Niccolo-Ajroldi/plainLM  
- Arithmetic tasks adapted from https://github.com/automl/DeltaProduct  
- Word problem tasks adapted from https://github.com/jopetty/word-problem  
- Copy task adapted from https://github.com/sjelassi/transformers_ssm_copy  

---

## Citation

If you find this code useful, please cite:

```bibtex
@inproceedings{
    movahedi2025fixedpoint,
    title={Fixed-Point {RNN}s: Interpolating from Diagonal to Dense},
    author={Sajad Movahedi and Felix Sarnthein and Nicola Muca Cirone and Antonio Orvieto},
    booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
    year={2025},
    url={https://openreview.net/forum?id=KT8y9pFgJE}
}
```
