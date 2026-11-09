# EveryO — オープンなニューラルコンピューティング

**言語：**日本語 · [English](../../README.md) · [すべての言語](../../README.md#documentation-in-10-languages)

EveryO は、ニューラルネットワークの研究、教育、再現可能な実験のためのオープンソース Python フレームワークです。NumPy ベースのコアには、テンソル、自動微分、ニューラルネットワーク層、オプティマイザー、学習機能が含まれます。CUDA、TensorFlow、ONNX はオプションです。

## クイックスタート

Python 3.9–3.12 が必要です。リポジトリのルートで実行します。

```bash
python -m pip install -e .
everyo doctor
```

## 実行できる例

```python
import everyo as eo
x = eo.tensor([2.0], requires_grad=True)
y = x * x
y.backward()
print(x.grad)
```

**期待される出力：**

```text
[4.]
```

これは `x = 2` における `x²` の微分値です。モデルと学習の例は [examples](../../examples/) をご覧ください。

## 技術的な範囲

- **コア：**NumPy、自動微分、層、最適化、シリアライズ。
- **検証：**回帰テスト、数値検証、CPU ベンチマーク。
- **統合：**オプションの CUDA、TensorFlow 比較、ONNX エクスポート。
- **制限：**独立した本番環境認証は未取得です。[評価レポート](../production-qualification-2026-11-09.md)を参照してください。

技術的な正式情報は[英語版](../../README.md)を基準とします。
