.PHONY: dist
dist:
	python setup.py sdist

.PHONY: upload
upload: dist
	twine upload dist/*

.PHONY: clean
clean:
	@rm -rf build dist *.egg-info

.PHONY: test
test:
	PYTHONPATH=. python test/test.py

.PHONY: test_decorator
test_decorator:
	PYTHONPATH=. python test/test_decorator.py