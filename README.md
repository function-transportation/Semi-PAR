## Usage
```python
python fixmatch_solider.py --train_label_file (訓練ファイルパス) --eval_label_file (評価用ファイルパス) --unlabel_label_file (訓練ファイルパスと同じで良い)
```

## 注意点
* 82行目のモデルパスを変更する or argsから渡すようにコードを変更
* 訓練ファイルと評価用ファイルは分ける
* 半教師あり学習も可能だがバイアスが拡大しがちなことから非推奨
