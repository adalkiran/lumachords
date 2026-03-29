uv tool install twine
rm -rf dist/ && uv build && twine upload --repository testpypi dist/* --config-file ~/.pypirc --verbose