# PyPI 發布指南

這份指南是針對目前這個 repo 的發布流程。發到 PyPI 之後，使用者會用：

```bash
python3 -m pip install daam-timm-vit
```

Python 程式裡的匯入名稱仍然是：

```python
from daam import TimmViTDAAM
```

## 目前狀態

- `pyproject.toml` 已經可以用 `setuptools` 建立 package。
- PyPI 發行名稱是 `daam-timm-vit`。
- Python import package 是 `daam`。
- `dist/`、`build/`、`*.egg-info/` 和 `.pypirc` 已經在 `.gitignore` 裡。
- `LICENSE` 是 MIT。
- `README.md` 裡給 PyPI 使用的圖片和文件連結已改成 GitHub 絕對 URL。

注意：PyPI 上已經有別人的 `daam` 發行名稱，所以不要把 `[project].name` 改成 `daam`。目前的 `daam-timm-vit` 比較安全。不過因為 import package 仍叫 `daam`，如果使用者同時安裝另一個也提供 `daam` package 的專案，可能會互相覆蓋。若你想完全避開這個風險，應該在第一次正式發布前把 `daam/` 目錄改名，例如 `daam_timm_vit/`，但這會改變使用者的 import API。

## 第一次發布前檢查

1. 確認 GitHub repo 已公開，否則 PyPI metadata 裡的 Homepage/Repository/Issues 連結會是 404。
2. 確認 `README.md` 在 PyPI 上也能讀懂。封面圖和文件連結已使用 GitHub 絕對 URL；發布前只要確認 repo 是公開的即可。

3. 確認版本號。第一次可以用 `0.1.0`；之後每次發布都要增加版本號。PyPI 不允許覆蓋已發布的同版本檔案。
4. 確認 runtime dependencies。這個套件會安裝 `torch`、`torchvision`、`timm`、`pillow`、`numpy`。如果你希望使用者自己選 CUDA/CPU 版 PyTorch，可以考慮未來把 `torch` 和 `torchvision` 移到 optional dependency，但目前設定是最容易 `pip install` 後直接使用的版本。

## 建置 package

在這台機器上建議用已安裝的 `uv`，因為系統 Python 是 externally managed，而且目前缺少 `python3.14-venv`：

```bash
rm -rf dist build *.egg-info
uv build
uvx --from twine twine check dist/*
```

如果你在另一台機器上有正常的 virtualenv，也可以用標準 PyPA 指令：

```bash
python3 -m pip install --upgrade build twine
python3 -m build
python3 -m twine check dist/*
```

成功後應該會產生兩種檔案：

```text
dist/daam_timm_vit-0.1.0.tar.gz
dist/daam_timm_vit-0.1.0-py3-none-any.whl
```

可以檢查 wheel 裡是否有你的 package：

```bash
python3 -m zipfile -l dist/daam_timm_vit-0.1.0-py3-none-any.whl | sed -n '1,120p'
```

## 先發到 TestPyPI

1. 到 https://test.pypi.org/account/register/ 註冊 TestPyPI 帳號。
2. 到 https://test.pypi.org/manage/account/#api-tokens 建立 API token。
3. 上傳到 TestPyPI：

   ```bash
   uvx --from twine twine upload --repository testpypi dist/*
   ```

4. Twine 問帳號密碼時：

   ```text
   username: __token__
   password: pypi-你的TestPyPI-token
   ```

5. 測試從 TestPyPI 安裝。因為 dependencies 在正式 PyPI，不一定都在 TestPyPI，所以加上 `--extra-index-url`：

   ```bash
   uv venv /tmp/daam-testpypi --clear
   uv pip install --python /tmp/daam-testpypi/bin/python \
     --index-url https://test.pypi.org/simple/ \
     --extra-index-url https://pypi.org/simple/ \
     daam-timm-vit==0.1.0
   ```

6. 做最小 import 測試：

   ```bash
   /tmp/daam-testpypi/bin/python - <<'PY'
   import importlib.metadata as metadata
   import daam

   print(metadata.version("daam-timm-vit"))
   print(daam.TimmViTDAAM)
   PY
   ```

如果 PyTorch 在你的平台上需要特殊 CUDA wheel，請先依照 PyTorch 官方方式安裝 `torch` / `torchvision`，再安裝這個套件。

## 發到正式 PyPI

TestPyPI 沒問題後：

1. 到 https://pypi.org/account/register/ 註冊正式 PyPI 帳號。
2. 到 https://pypi.org/manage/account/#api-tokens 建立 API token。
3. 建議重新清乾淨並 build 一次：

   ```bash
   rm -rf dist build *.egg-info
   uv build
   uvx --from twine twine check dist/*
   ```

4. 上傳正式 PyPI：

   ```bash
   uvx --from twine twine upload dist/*
   ```

5. Twine 問帳號密碼時：

   ```text
   username: __token__
   password: pypi-你的正式PyPI-token
   ```

6. 發布完成後測試：

   ```bash
   uv venv /tmp/daam-pypi --clear
   uv pip install --python /tmp/daam-pypi/bin/python daam-timm-vit
   /tmp/daam-pypi/bin/python - <<'PY'
   import daam
   print(daam.TimmViTDAAM)
   PY
   ```

## 之後每次更新

1. 修改程式和 README。
2. 在 `pyproject.toml` 增加 `version`，例如 `0.1.0` -> `0.1.1`。
3. commit 並打 tag：

   ```bash
   git add pyproject.toml README.md daam docs
   git commit -m "Release 0.1.1"
   git tag v0.1.1
   git push
   git push origin v0.1.1
   ```

4. 重新 build、`twine check`、先 TestPyPI、再正式 PyPI。

## 常見錯誤

- `File already exists`：同版本已經上傳過，改 `version` 後重 build。
- `The name ... is too similar to an existing project`：PyPI 認為名稱不可用，只能換 `[project].name`。
- README 圖片沒顯示：把相對路徑圖片改成公開的絕對 URL。
- `No matching distribution found`：確認版本號、TestPyPI/PyPI index、Python 版本是否符合 `requires-python = ">=3.10"`。
- 安裝 PyTorch 很慢或 CUDA 不對：先依你的 CUDA/CPU 環境安裝 PyTorch，再安裝 `daam-timm-vit`。
