setup:
	echo "Creating virtual environment."
	python3 -m venv .venv
	echo "Virtual environment created."
	echo "Installing requirements"
	.venv/bin/activate; python3 -m pip install -r requirements.txt
	echo "Requirements installed."
