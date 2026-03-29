uv tool install twine
rm -rf dist/ && uv build && twine upload dist/* --config-file ~/.pypirc