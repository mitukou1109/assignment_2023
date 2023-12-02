# assignment_2023（ロボットインテリジェンス課題）

## 実行環境
- Windows 11
- Python 3.11.6 (@ pipenv)

## 実行方法
1. pipenv等で依存パッケージをインストール
```
$ pipenv sync                        # pipenvの場合
$ pip install -r requirements.txt    # その他
```
2. assignment_2023/\__main\__.pyでパラメータを設定
  - `epochs`：学習終了までのエポック数
  - `batch_size`：バッチサイズ
  - `hidden_layer_features`：中間層のニューロン数（任意の層数に対応，1層の場合でもlistで指定） 
  - `learning_rate`：学習率
  - `noise_prob`：ノイズの生起確率（0～1）
  - `seed`：乱数のシード値
  - `train`：`True`の場合学習を行い，`False`の場合テストのみ実行
  - `show_data_sample`：学習に用いるデータのサンプルを表示
  - `show_optimal_stimuli`：テスト終了後に最適刺激を表示
  - `show_receptive_field`：テスト終了後に受容野を表示
3. assignment_2023を実行
```
$ python -m assignment_2023
```
4. visualizer.pyを実行（損失関数の推移と学習曲線を表示）
```
$ python assignment_2023/visualizer.py
```

## 補足事項
- データセットは初回実行時dataディレクトリにダウンロードされます．
- 学習時は各エポックが終了するたびcheckpointディレクトリにパラメータを記録したチェックポイント（npzファイル），logディレクトリに学習ログ（csvファイル）が保存されます．
  - 学習を中断しても，assignment_2023の実行時に引数としてチェックポイントファイルへのパスを渡せばそれを読み込んで再開できます．
- visualizer.pyは引数にログファイルへのパスを指定できます．指定しなかった場合は最新のログを参照します．
- モジュール・クラス名はPyTorchに倣っています．
