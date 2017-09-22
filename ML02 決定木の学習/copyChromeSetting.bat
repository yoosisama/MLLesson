@echo "jupyter_notebook_config.py を ~/.jupyter  にコピーし，jupyter notebook を起動します"
@echo "ファイルが用意できたことを確認してから何かキーを押してください"
pause
copy jupyter_notebook_config.py %USERPROFILE%\.jupyter
jupyter notebook
