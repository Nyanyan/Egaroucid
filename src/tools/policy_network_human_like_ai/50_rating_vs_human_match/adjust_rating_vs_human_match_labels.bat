@echo off
rem Open the current rating/agreement figures for drag-and-save label editing.
call "%~dp0run_rating_vs_human_match.bat" --adjust-labels %*
exit /b %ERRORLEVEL%
